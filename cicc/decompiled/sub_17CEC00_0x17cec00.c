// Function: sub_17CEC00
// Address: 0x17cec00
//
__int64 __fastcall sub_17CEC00(__int64 *a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r14
  bool v8; // cc
  _QWORD *v9; // r12
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // r10
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 *v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  int v21; // [rsp+Ch] [rbp-84h]
  __int64 v22; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+18h] [rbp-78h]
  __int64 v25[2]; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v26; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v27[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v28; // [rsp+50h] [rbp-40h]

  v5 = a2;
  v8 = a3[16] <= 0x10u;
  v25[0] = a4;
  if ( v8 && *(_BYTE *)(a4 + 16) <= 0x10u )
  {
    v26 = (__int64 *)a4;
    v27[4] = 0;
    return sub_15A2E80(a2, (__int64)a3, &v26, 1u, 1u, (__int64)v27, 0);
  }
  else
  {
    v28 = 257;
    if ( !a2 )
    {
      v20 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v20 = **(_QWORD **)(v20 + 16);
      v5 = *(_QWORD *)(v20 + 24);
    }
    v11 = sub_1648A60(72, 2u);
    v9 = v11;
    if ( v11 )
    {
      v23 = (__int64)v11;
      v22 = (__int64)(v11 - 6);
      v12 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v12 = **(_QWORD **)(v12 + 16);
      v21 = *(_DWORD *)(v12 + 8) >> 8;
      v13 = (__int64 *)sub_15F9F50(v5, (__int64)v25, 1);
      v14 = (__int64 *)sub_1646BA0(v13, v21);
      v15 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 || (v15 = *(_QWORD *)v25[0], *(_BYTE *)(*(_QWORD *)v25[0] + 8LL) == 16) )
        v14 = sub_16463B0(v14, *(_QWORD *)(v15 + 32));
      sub_15F1EA0((__int64)v9, (__int64)v14, 32, v22, 2, 0);
      v9[7] = v5;
      v9[8] = sub_15F9F50(v5, (__int64)v25, 1);
      sub_15F9CE0((__int64)v9, (__int64)a3, v25, 1, (__int64)v27);
    }
    else
    {
      v23 = 0;
    }
    sub_15FA2E0((__int64)v9, 1);
    v16 = a1[1];
    if ( v16 )
    {
      v17 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v16 + 40, (__int64)v9);
      v18 = v9[3];
      v19 = *v17;
      v9[4] = v17;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v19 | v18 & 7;
      *(_QWORD *)(v19 + 8) = v9 + 3;
      *v17 = *v17 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780(v23, a5);
    sub_12A86E0(a1, (__int64)v9);
  }
  return (__int64)v9;
}
