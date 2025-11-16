// Function: sub_12A8800
// Address: 0x12a8800
//
__int64 __fastcall sub_12A8800(__int64 *a1, __int64 a2, _BYTE *a3, unsigned int a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  bool v11; // cc
  _QWORD *v12; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int64 *v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // [rsp+4h] [rbp-7Ch]
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+20h] [rbp-60h] BYREF
  __int64 v31; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v32[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  v7 = a2;
  v9 = sub_1643350(a1[3]);
  v10 = sub_159C470(v9, a4, 0);
  v11 = a3[16] <= 0x10u;
  v30 = v10;
  if ( v11 )
  {
    v32[4] = 0;
    return sub_15A2E80(a2, (_DWORD)a3, (unsigned int)&v30, 1, 1, (unsigned int)v32, 0);
  }
  else
  {
    v33 = 257;
    if ( !a2 )
    {
      v25 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v25 = **(_QWORD **)(v25 + 16);
      v7 = *(_QWORD *)(v25 + 24);
    }
    v14 = sub_1648A60(72, 2);
    v12 = (_QWORD *)v14;
    if ( v14 )
    {
      v28 = v14;
      v27 = v14 - 48;
      v15 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v15 = **(_QWORD **)(v15 + 16);
      v26 = *(_DWORD *)(v15 + 8) >> 8;
      v16 = sub_15F9F50(v7, &v30, 1);
      v17 = sub_1646BA0(v16, v26);
      v18 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 || (v18 = *(_QWORD *)v30, *(_BYTE *)(*(_QWORD *)v30 + 8LL) == 16) )
        v17 = sub_16463B0(v17, *(_QWORD *)(v18 + 32));
      sub_15F1EA0(v12, v17, 32, v27, 2, 0);
      v12[7] = v7;
      v12[8] = sub_15F9F50(v7, &v30, 1);
      sub_15F9CE0(v12, a3, &v30, 1, v32);
    }
    else
    {
      v28 = 0;
    }
    sub_15FA2E0(v12, 1);
    v19 = a1[1];
    if ( v19 )
    {
      v20 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v19 + 40, v12);
      v21 = v12[3];
      v22 = *v20;
      v12[4] = v20;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      v12[3] = v22 | v21 & 7;
      *(_QWORD *)(v22 + 8) = v12 + 3;
      *v20 = *v20 & 7 | (unsigned __int64)(v12 + 3);
    }
    sub_164B780(v28, a5);
    v23 = *a1;
    if ( *a1 )
    {
      v31 = *a1;
      sub_1623A60(&v31, v23, 2);
      if ( v12[6] )
        sub_161E7C0(v12 + 6);
      v24 = v31;
      v12[6] = v31;
      if ( v24 )
        sub_1623210(&v31, v24, v12 + 6);
    }
  }
  return (__int64)v12;
}
