// Function: sub_128B460
// Address: 0x128b460
//
__int64 __fastcall sub_128B460(__int64 *a1, __int64 a2, _BYTE *a3, __int64 **a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r10
  __int64 **v19; // rcx
  __int64 **v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 *v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // [rsp+8h] [rbp-88h]
  unsigned int v33; // [rsp+Ch] [rbp-84h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v38[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v39; // [rsp+50h] [rbp-40h]

  v7 = a2;
  if ( a3[16] <= 0x10u )
  {
    if ( !a5 )
    {
LABEL_29:
      v38[4] = 0;
      return sub_15A2E80(a2, (_DWORD)a3, (_DWORD)a4, a5, 1, (unsigned int)v38, 0);
    }
    v10 = 0;
    while ( *((_BYTE *)a4[v10] + 16) <= 0x10u )
    {
      if ( ++v10 == a5 )
        goto LABEL_29;
    }
  }
  v11 = a5 + 1;
  v39 = 257;
  if ( !a2 )
  {
    v30 = *(_QWORD *)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      v30 = **(_QWORD **)(v30 + 16);
    v7 = *(_QWORD *)(v30 + 24);
  }
  v12 = sub_1648A60(72, v11);
  v13 = (_QWORD *)v12;
  if ( v12 )
  {
    v34 = v12;
    v14 = *(_QWORD *)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      v14 = **(_QWORD **)(v14 + 16);
    v32 = a5 + 1;
    v33 = *(_DWORD *)(v14 + 8) >> 8;
    v15 = sub_15F9F50(v7, a4, a5);
    v16 = sub_1646BA0(v15, v33);
    v17 = (unsigned int)(a5 + 1);
    v18 = v16;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v31 = sub_16463B0(v16, *(_QWORD *)(*(_QWORD *)a3 + 32LL));
      v17 = v32;
      v18 = v31;
    }
    else
    {
      v19 = &a4[a5];
      if ( v19 != a4 )
      {
        v20 = a4;
        while ( 1 )
        {
          v21 = **v20;
          if ( *(_BYTE *)(v21 + 8) == 16 )
            break;
          if ( v19 == ++v20 )
            goto LABEL_16;
        }
        v22 = sub_16463B0(v18, *(_QWORD *)(v21 + 32));
        v17 = v32;
        v18 = v22;
      }
    }
LABEL_16:
    sub_15F1EA0(v13, v18, 32, &v13[-3 * (unsigned int)(a5 + 1)], v17, 0);
    v13[7] = v7;
    v13[8] = sub_15F9F50(v7, a4, a5);
    sub_15F9CE0(v13, a3, a4, a5, v38);
  }
  else
  {
    v34 = 0;
  }
  sub_15FA2E0(v13, 1);
  v23 = a1[1];
  if ( v23 )
  {
    v24 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v23 + 40, v13);
    v25 = v13[3];
    v26 = *v24;
    v13[4] = v24;
    v26 &= 0xFFFFFFFFFFFFFFF8LL;
    v13[3] = v26 | v25 & 7;
    *(_QWORD *)(v26 + 8) = v13 + 3;
    *v24 = *v24 & 7 | (unsigned __int64)(v13 + 3);
  }
  sub_164B780(v34, a6);
  v27 = *a1;
  if ( *a1 )
  {
    v37 = *a1;
    sub_1623A60(&v37, v27, 2);
    if ( v13[6] )
      sub_161E7C0(v13 + 6);
    v28 = v37;
    v13[6] = v37;
    if ( v28 )
      sub_1623210(&v37, v28, v13 + 6);
  }
  return (__int64)v13;
}
