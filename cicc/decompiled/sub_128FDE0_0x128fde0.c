// Function: sub_128FDE0
// Address: 0x128fde0
//
__int64 __fastcall sub_128FDE0(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v5; // rax
  __int64 v6; // rdx
  char *v7; // r13
  char v8; // al
  __int64 *v9; // rbx
  __int64 v10; // r12
  int v11; // eax
  __int64 *v13; // rbx
  __int64 *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // r13d
  __int64 v18; // rdi
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 *v26; // r13
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // r13
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v39[2]; // [rsp+10h] [rbp-60h] BYREF
  char v40; // [rsp+20h] [rbp-50h]
  char v41; // [rsp+21h] [rbp-4Fh]
  _BYTE v42[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v43; // [rsp+40h] [rbp-30h]

  v5 = sub_128D0F0(a1, a2[9], a3, a4, a5);
  v6 = *(_QWORD *)v5;
  v7 = v5;
  v8 = *(_BYTE *)(*(_QWORD *)v5 + 8LL);
  if ( v8 == 16 )
    v8 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  if ( (unsigned __int8)(v8 - 1) <= 5u )
  {
    v41 = 1;
    v9 = a1[1];
    v39[0] = "neg";
    v40 = 3;
    if ( (unsigned __int8)v7[16] > 0x10u )
    {
      v43 = 257;
      v15 = sub_15FB5B0(v7, v42, 0);
      v16 = v9[4];
      v17 = *((_DWORD *)v9 + 10);
      v10 = v15;
      if ( v16 )
        sub_1625C10(v15, 3, v16);
      sub_15F2440(v10, v17);
      v18 = v9[1];
      if ( v18 )
      {
        v19 = (__int64 *)v9[2];
        sub_157E9D0(v18 + 40, v10);
        v20 = *(_QWORD *)(v10 + 24);
        v21 = *v19;
        *(_QWORD *)(v10 + 32) = v19;
        v21 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v21 | v20 & 7;
        *(_QWORD *)(v21 + 8) = v10 + 24;
        *v19 = *v19 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v39);
      v22 = *v9;
      if ( *v9 )
      {
        v38 = *v9;
        sub_1623A60(&v38, v22, 2);
        if ( *(_QWORD *)(v10 + 48) )
          sub_161E7C0(v10 + 48);
        v23 = v38;
        *(_QWORD *)(v10 + 48) = v38;
        if ( v23 )
          sub_1623210(&v38, v23, v10 + 48);
      }
    }
    else
    {
      v10 = sub_15A2BF0(v7);
    }
    if ( unk_4D04700 )
    {
      if ( *(_BYTE *)(v10 + 16) > 0x17u )
      {
        v11 = sub_15F24E0(v10);
        sub_15F2440(v10, v11 | 1u);
      }
    }
    return v10;
  }
  if ( (unsigned __int8)sub_127B3A0(*a2) )
  {
    v41 = 1;
    v13 = a1[1];
    v39[0] = "neg";
    v40 = 3;
    if ( (unsigned __int8)v7[16] > 0x10u )
    {
      v43 = 257;
      v31 = sub_15FB530(v7, v42, 0);
      v32 = v13[1];
      v10 = v31;
      if ( v32 )
      {
        v33 = (__int64 *)v13[2];
        sub_157E9D0(v32 + 40, v31);
        v34 = *(_QWORD *)(v10 + 24);
        v35 = *v33;
        *(_QWORD *)(v10 + 32) = v33;
        v35 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v35 | v34 & 7;
        *(_QWORD *)(v35 + 8) = v10 + 24;
        *v33 = *v33 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v39);
      v36 = *v13;
      if ( *v13 )
      {
        v38 = *v13;
        sub_1623A60(&v38, v36, 2);
        if ( *(_QWORD *)(v10 + 48) )
          sub_161E7C0(v10 + 48);
        v37 = v38;
        *(_QWORD *)(v10 + 48) = v38;
        if ( v37 )
          sub_1623210(&v38, v37, v10 + 48);
      }
      sub_15F2330(v10, 1);
      return v10;
    }
    return sub_15A2B90(v7, 0, 1);
  }
  else
  {
    v41 = 1;
    v14 = a1[1];
    v39[0] = "neg";
    v40 = 3;
    if ( (unsigned __int8)v7[16] > 0x10u )
    {
      v43 = 257;
      v24 = sub_15FB530(v7, v42, 0);
      v25 = v14[1];
      v10 = v24;
      if ( v25 )
      {
        v26 = (__int64 *)v14[2];
        sub_157E9D0(v25 + 40, v24);
        v27 = *(_QWORD *)(v10 + 24);
        v28 = *v26;
        *(_QWORD *)(v10 + 32) = v26;
        v28 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v28 | v27 & 7;
        *(_QWORD *)(v28 + 8) = v10 + 24;
        *v26 = *v26 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v39);
      v29 = *v14;
      if ( *v14 )
      {
        v38 = *v14;
        sub_1623A60(&v38, v29, 2);
        if ( *(_QWORD *)(v10 + 48) )
          sub_161E7C0(v10 + 48);
        v30 = v38;
        *(_QWORD *)(v10 + 48) = v38;
        if ( v30 )
          sub_1623210(&v38, v30, v10 + 48);
      }
      return v10;
    }
    return sub_15A2B90(v7, 0, 0);
  }
}
