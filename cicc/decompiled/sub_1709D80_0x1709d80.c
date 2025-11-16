// Function: sub_1709D80
// Address: 0x1709d80
//
__int64 *__fastcall sub_1709D80(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // r13
  bool v8; // cc
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // r10
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 *v18; // r14
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  bool v22; // zf
  __int64 *v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  int v27; // [rsp+Ch] [rbp-84h]
  __int64 v28; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+18h] [rbp-78h]
  __int64 v31[2]; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v32; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v33[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v34; // [rsp+50h] [rbp-40h]

  v6 = a2;
  v8 = a3[16] <= 0x10u;
  v31[0] = a4;
  if ( v8 && *(_BYTE *)(a4 + 16) <= 0x10u )
  {
    v32 = (__int64 *)a4;
    v33[4] = 0;
    v9 = (__int64 *)sub_15A2E80(a2, (__int64)a3, &v32, 1u, 1u, (__int64)v33, 0);
    v10 = sub_14DBA30((__int64)v9, *(_QWORD *)(a1 + 96), 0);
    if ( v10 )
      return (__int64 *)v10;
  }
  else
  {
    v34 = 257;
    if ( !a2 )
    {
      v26 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v26 = **(_QWORD **)(v26 + 16);
      v6 = *(_QWORD *)(v26 + 24);
    }
    v12 = sub_1648A60(72, 2u);
    v9 = v12;
    if ( v12 )
    {
      v29 = (__int64)v12;
      v28 = (__int64)(v12 - 6);
      v13 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v13 = **(_QWORD **)(v13 + 16);
      v27 = *(_DWORD *)(v13 + 8) >> 8;
      v14 = (__int64 *)sub_15F9F50(v6, (__int64)v31, 1);
      v15 = (__int64 *)sub_1646BA0(v14, v27);
      v16 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 || (v16 = *(_QWORD *)v31[0], *(_BYTE *)(*(_QWORD *)v31[0] + 8LL) == 16) )
        v15 = sub_16463B0(v15, *(_QWORD *)(v16 + 32));
      sub_15F1EA0((__int64)v9, (__int64)v15, 32, v28, 2, 0);
      v9[7] = v6;
      v9[8] = sub_15F9F50(v6, (__int64)v31, 1);
      sub_15F9CE0((__int64)v9, (__int64)a3, v31, 1, (__int64)v33);
    }
    else
    {
      v29 = 0;
    }
    sub_15FA2E0((__int64)v9, 1);
    v17 = *(_QWORD *)(a1 + 8);
    if ( v17 )
    {
      v18 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, (__int64)v9);
      v19 = v9[3];
      v20 = *v18;
      v9[4] = (__int64)v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v9 + 3;
      *v18 = *v18 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780(v29, a5);
    v22 = *(_QWORD *)(a1 + 80) == 0;
    v32 = v9;
    if ( v22 )
      sub_4263D6(v29, a5, v21);
    (*(void (__fastcall **)(__int64, __int64 **))(a1 + 88))(a1 + 64, &v32);
    v23 = *(__int64 **)a1;
    if ( *(_QWORD *)a1 )
    {
      v32 = *(__int64 **)a1;
      sub_1623A60((__int64)&v32, (__int64)v23, 2);
      v24 = v9[6];
      if ( v24 )
        sub_161E7C0((__int64)(v9 + 6), v24);
      v25 = (unsigned __int8 *)v32;
      v9[6] = (__int64)v32;
      if ( v25 )
        sub_1623210((__int64)&v32, v25, (__int64)(v9 + 6));
    }
  }
  return v9;
}
