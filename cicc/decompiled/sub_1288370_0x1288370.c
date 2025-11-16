// Function: sub_1288370
// Address: 0x1288370
//
__int64 __fastcall sub_1288370(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  char v5; // al
  bool v6; // cc
  __int64 *v7; // rbx
  __int64 v8; // r12
  int v9; // eax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r13d
  __int64 v14; // rdi
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 *v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v38[2]; // [rsp+10h] [rbp-70h] BYREF
  char v39; // [rsp+20h] [rbp-60h]
  char v40; // [rsp+21h] [rbp-5Fh]
  _QWORD v41[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v42; // [rsp+40h] [rbp-40h]

  v5 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v5 == 16 )
    v5 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  if ( (unsigned __int8)(v5 - 1) <= 5u )
  {
    v6 = a2[16] <= 0x10u;
    v40 = 1;
    v38[0] = "sub";
    v7 = *(__int64 **)(a1 + 8);
    v39 = 3;
    if ( !v6 || *(_BYTE *)(a3 + 16) > 0x10u || (v8 = sub_15A2A30(14, a2, a3, 0, 0)) == 0 )
    {
      v42 = 257;
      v11 = sub_15FB440(14, a2, a3, v41, 0);
      v12 = v7[4];
      v13 = *((_DWORD *)v7 + 10);
      v8 = v11;
      if ( v12 )
        sub_1625C10(v11, 3, v12);
      sub_15F2440(v8, v13);
      v14 = v7[1];
      if ( v14 )
      {
        v15 = (__int64 *)v7[2];
        sub_157E9D0(v14 + 40, v8);
        v16 = *(_QWORD *)(v8 + 24);
        v17 = *v15;
        *(_QWORD *)(v8 + 32) = v15;
        v17 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v17 | v16 & 7;
        *(_QWORD *)(v17 + 8) = v8 + 24;
        *v15 = *v15 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v38);
      v18 = *v7;
      if ( *v7 )
      {
        v41[0] = *v7;
        sub_1623A60(v41, v18, 2);
        if ( *(_QWORD *)(v8 + 48) )
          sub_161E7C0(v8 + 48);
        v19 = v41[0];
        *(_QWORD *)(v8 + 48) = v41[0];
        if ( v19 )
          sub_1623210(v41, v19, v8 + 48);
      }
    }
    if ( unk_4D04700 && *(_BYTE *)(v8 + 16) > 0x17u )
    {
      v9 = sub_15F24E0(v8);
      sub_15F2440(v8, v9 | 1u);
    }
    return v8;
  }
  if ( (unsigned __int8)sub_127B3A0(a4) )
  {
    v6 = a2[16] <= 0x10u;
    v40 = 1;
    v38[0] = "sub";
    v20 = *(__int64 **)(a1 + 8);
    v39 = 3;
    if ( !v6 || (v21 = 1, *(_BYTE *)(a3 + 16) > 0x10u) )
    {
      v42 = 257;
      v22 = sub_15FB440(13, a2, a3, v41, 0);
      v23 = v20[1];
      v8 = v22;
      if ( v23 )
      {
        v24 = (__int64 *)v20[2];
        sub_157E9D0(v23 + 40, v22);
        v25 = *(_QWORD *)(v8 + 24);
        v26 = *v24;
        *(_QWORD *)(v8 + 32) = v24;
        v26 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v26 | v25 & 7;
        *(_QWORD *)(v26 + 8) = v8 + 24;
        *v24 = *v24 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v38);
      v27 = *v20;
      if ( *v20 )
      {
        v37 = *v20;
        sub_1623A60(&v37, v27, 2);
        if ( *(_QWORD *)(v8 + 48) )
          sub_161E7C0(v8 + 48);
        v28 = v37;
        *(_QWORD *)(v8 + 48) = v37;
        if ( v28 )
          sub_1623210(&v37, v28, v8 + 48);
      }
      sub_15F2330(v8, 1);
      return v8;
    }
    return sub_15A2B60(a2, a3, 0, v21);
  }
  v6 = a2[16] <= 0x10u;
  v40 = 1;
  v38[0] = "sub";
  v29 = *(__int64 **)(a1 + 8);
  v39 = 3;
  if ( v6 && *(_BYTE *)(a3 + 16) <= 0x10u )
  {
    v21 = 0;
    return sub_15A2B60(a2, a3, 0, v21);
  }
  v42 = 257;
  v30 = sub_15FB440(13, a2, a3, v41, 0);
  v31 = v29[1];
  v8 = v30;
  if ( v31 )
  {
    v32 = (__int64 *)v29[2];
    sub_157E9D0(v31 + 40, v30);
    v33 = *(_QWORD *)(v8 + 24);
    v34 = *v32;
    *(_QWORD *)(v8 + 32) = v32;
    v34 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v34 | v33 & 7;
    *(_QWORD *)(v34 + 8) = v8 + 24;
    *v32 = *v32 & 7 | (v8 + 24);
  }
  sub_164B780(v8, v38);
  v35 = *v29;
  if ( *v29 )
  {
    v37 = *v29;
    sub_1623A60(&v37, v35, 2);
    if ( *(_QWORD *)(v8 + 48) )
      sub_161E7C0(v8 + 48);
    v36 = v37;
    *(_QWORD *)(v8 + 48) = v37;
    if ( v36 )
      sub_1623210(&v37, v36, v8 + 48);
  }
  return v8;
}
