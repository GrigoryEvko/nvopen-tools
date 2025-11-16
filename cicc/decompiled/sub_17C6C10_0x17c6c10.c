// Function: sub_17C6C10
// Address: 0x17c6c10
//
unsigned __int64 __fastcall sub_17C6C10(__int64 a1)
{
  signed __int64 v2; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v4; // r13
  __int64 *v5; // rax
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // r13
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int8 *v20; // rsi
  __int64 *v21; // rax
  __int64 v22; // r14
  __int64 v23; // r13
  _QWORD *v24; // rax
  _QWORD *v25; // r12
  char v26; // al
  __int64 v27; // [rsp+0h] [rbp-C0h]
  __int64 v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+8h] [rbp-B8h]
  _QWORD *v30; // [rsp+8h] [rbp-B8h]
  __int64 v31; // [rsp+8h] [rbp-B8h]
  unsigned __int64 *v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v34[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v35; // [rsp+30h] [rbp-90h]
  __int64 *v36; // [rsp+40h] [rbp-80h] BYREF
  __int64 v37; // [rsp+48h] [rbp-78h]
  unsigned __int64 *v38; // [rsp+50h] [rbp-70h]
  __int64 v39; // [rsp+58h] [rbp-68h]
  __int64 v40; // [rsp+60h] [rbp-60h]
  int v41; // [rsp+68h] [rbp-58h]
  __int64 v42; // [rsp+70h] [rbp-50h]
  __int64 v43; // [rsp+78h] [rbp-48h]

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    v21 = (__int64 *)sub_15996B0(**(_QWORD **)(a1 + 40), *(char **)(a1 + 8), v2, 1);
    v22 = *v21;
    v23 = (__int64)v21;
    v36 = (__int64 *)"__llvm_profile_filename";
    LOWORD(v38) = 259;
    v24 = sub_1648A60(88, 1u);
    v25 = v24;
    if ( v24 )
      sub_15E51E0((__int64)v24, *(_QWORD *)(a1 + 40), v22, 1, 4, v23, (__int64)&v36, 0, 0, 0, 0);
    if ( *(_DWORD *)(a1 + 100) != 3 )
    {
      v26 = *((_BYTE *)v25 + 32);
      *((_BYTE *)v25 + 32) = v26 & 0xF0;
      if ( (v26 & 0x30) != 0 )
        *((_BYTE *)v25 + 33) |= 0x40u;
      v25[6] = sub_1633B90(*(_QWORD *)(a1 + 40), "__llvm_profile_filename", 0x17u);
    }
  }
  result = sub_16321A0(*(_QWORD *)(a1 + 40), (__int64)"__llvm_profile_register_functions", 33);
  v4 = result;
  if ( result )
  {
    v5 = (__int64 *)sub_1643270(**(_QWORD ***)(a1 + 40));
    v6 = *(_QWORD *)(a1 + 40);
    v36 = v34;
    v34[1] = 19;
    v27 = v6;
    LOWORD(v38) = 261;
    v34[0] = (__int64)"__llvm_profile_init";
    v28 = sub_16453E0(v5, 0);
    v7 = sub_1648B60(120);
    v8 = v7;
    if ( v7 )
      sub_15E2490(v7, v28, 7, (__int64)&v36, v27);
    *(_BYTE *)(v8 + 32) = *(_BYTE *)(v8 + 32) & 0x3F | 0x80;
    sub_15E0D50(v8, -1, 26);
    if ( *(_BYTE *)a1 )
      sub_15E0D50(v8, -1, 28);
    v9 = *(__int64 **)(a1 + 40);
    v35 = 257;
    v29 = *v9;
    v10 = (_QWORD *)sub_22077B0(64);
    v11 = (__int64)v10;
    if ( v10 )
    {
      v12 = v29;
      v30 = v10;
      sub_157FB60(v10, v12, (__int64)v34, v8, 0);
      v11 = (__int64)v30;
    }
    v13 = sub_157E9C0(v11);
    v36 = 0;
    v39 = v13;
    v37 = v11;
    v38 = (unsigned __int64 *)(v11 + 40);
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v35 = 257;
    sub_1285290((__int64 *)&v36, *(_QWORD *)(*(_QWORD *)v4 + 24LL), v4, 0, 0, (__int64)v34, 0);
    v35 = 257;
    v31 = v39;
    v14 = sub_1648A60(56, 0);
    v15 = v14;
    if ( v14 )
      sub_15F6F90((__int64)v14, v31, 0, 0);
    if ( v37 )
    {
      v32 = v38;
      sub_157E9D0(v37 + 40, (__int64)v15);
      v16 = *v32;
      v17 = v15[3] & 7LL;
      v15[4] = v32;
      v16 &= 0xFFFFFFFFFFFFFFF8LL;
      v15[3] = v16 | v17;
      *(_QWORD *)(v16 + 8) = v15 + 3;
      *v32 = *v32 & 7 | (unsigned __int64)(v15 + 3);
    }
    sub_164B780((__int64)v15, v34);
    if ( v36 )
    {
      v33 = (__int64)v36;
      sub_1623A60((__int64)&v33, (__int64)v36, 2);
      v18 = v15[6];
      v19 = (__int64)(v15 + 6);
      if ( v18 )
      {
        sub_161E7C0((__int64)(v15 + 6), v18);
        v19 = (__int64)(v15 + 6);
      }
      v20 = (unsigned __int8 *)v33;
      v15[6] = v33;
      if ( v20 )
        sub_1623210((__int64)&v33, v20, v19);
    }
    result = sub_1B28000(*(_QWORD *)(a1 + 40), v8, 0, 0);
    if ( v36 )
      return sub_161E7C0((__int64)&v36, (__int64)v36);
  }
  return result;
}
