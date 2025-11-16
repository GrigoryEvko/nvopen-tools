// Function: sub_1709730
// Address: 0x1709730
//
unsigned __int8 *__fastcall sub_1709730(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 **a4,
        unsigned __int64 a5,
        __int64 *a6)
{
  __int64 v7; // r14
  __int64 v10; // rax
  unsigned int v11; // r8d
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // rax
  int v17; // r8d
  __int64 *v18; // r10
  __int64 **v19; // rcx
  __int64 **v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rdi
  unsigned __int64 *v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rdx
  bool v28; // zf
  __int64 v29; // rsi
  __int64 v30; // rsi
  unsigned __int8 *v31; // rsi
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rax
  int v36; // [rsp+Ch] [rbp-84h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v41; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v42[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v43; // [rsp+50h] [rbp-40h]

  v7 = a2;
  if ( a3[16] <= 0x10u )
  {
    if ( !a5 )
    {
LABEL_30:
      v42[4] = 0;
      v13 = (unsigned __int8 *)sub_15A2E80(a2, (__int64)a3, a4, a5, 1u, (__int64)v42, 0);
      v35 = sub_14DBA30((__int64)v13, *(_QWORD *)(a1 + 96), 0);
      if ( v35 )
        return (unsigned __int8 *)v35;
      return v13;
    }
    v10 = 0;
    while ( *((_BYTE *)a4[v10] + 16) <= 0x10u )
    {
      if ( ++v10 == a5 )
        goto LABEL_30;
    }
  }
  v11 = a5 + 1;
  v43 = 257;
  if ( !a2 )
  {
    v33 = *(_QWORD *)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      v33 = **(_QWORD **)(v33 + 16);
    v7 = *(_QWORD *)(v33 + 24);
  }
  v12 = (unsigned __int8 *)sub_1648A60(72, v11);
  v13 = v12;
  if ( v12 )
  {
    v38 = (__int64)v12;
    v37 = (__int64)&v12[-24 * (unsigned int)(a5 + 1)];
    v14 = *(_QWORD *)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      v14 = **(_QWORD **)(v14 + 16);
    v36 = *(_DWORD *)(v14 + 8) >> 8;
    v15 = (__int64 *)sub_15F9F50(v7, (__int64)a4, a5);
    v16 = (__int64 *)sub_1646BA0(v15, v36);
    v17 = a5 + 1;
    v18 = v16;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v34 = sub_16463B0(v16, *(_QWORD *)(*(_QWORD *)a3 + 32LL));
      v17 = a5 + 1;
      v18 = v34;
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
        v17 = a5 + 1;
        v18 = v22;
      }
    }
LABEL_16:
    sub_15F1EA0((__int64)v13, (__int64)v18, 32, v37, v17, 0);
    *((_QWORD *)v13 + 7) = v7;
    *((_QWORD *)v13 + 8) = sub_15F9F50(v7, (__int64)a4, a5);
    sub_15F9CE0((__int64)v13, (__int64)a3, (__int64 *)a4, a5, (__int64)v42);
  }
  else
  {
    v38 = 0;
  }
  sub_15FA2E0((__int64)v13, 1);
  v23 = *(_QWORD *)(a1 + 8);
  if ( v23 )
  {
    v24 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v23 + 40, (__int64)v13);
    v25 = *((_QWORD *)v13 + 3);
    v26 = *v24;
    *((_QWORD *)v13 + 4) = v24;
    v26 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)v13 + 3) = v26 | v25 & 7;
    *(_QWORD *)(v26 + 8) = v13 + 24;
    *v24 = *v24 & 7 | (unsigned __int64)(v13 + 24);
  }
  sub_164B780(v38, a6);
  v28 = *(_QWORD *)(a1 + 80) == 0;
  v41 = v13;
  if ( v28 )
    sub_4263D6(v38, a6, v27);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v41);
  v29 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v41 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v41, v29, 2);
    v30 = *((_QWORD *)v13 + 6);
    if ( v30 )
      sub_161E7C0((__int64)(v13 + 48), v30);
    v31 = v41;
    *((_QWORD *)v13 + 6) = v41;
    if ( v31 )
      sub_1623210((__int64)&v41, v31, (__int64)(v13 + 48));
  }
  return v13;
}
