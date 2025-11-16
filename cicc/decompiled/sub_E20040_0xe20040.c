// Function: sub_E20040
// Address: 0xe20040
//
__int64 __fastcall sub_E20040(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // rax
  int v3; // r12d
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r13
  __int64 v13; // rax
  __int16 v14; // cx
  __int16 v15; // si
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // r12
  __int64 v20; // r15
  __int64 v21; // r9
  _BYTE *v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r14
  __int64 v28; // rax
  char v29; // al
  unsigned __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // r8
  unsigned __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r15
  __int64 v41; // rax
  char v42; // al
  __int64 v43; // [rsp+8h] [rbp-38h]

  v2 = (_BYTE *)*a1;
  if ( *a1 == a1[1] || *v2 != 85 )
  {
    v3 = sub_E0E0E0((__int64)a1);
    v11 = sub_E1AEA0((__int64)a1, a2, v4, v5, v6);
    if ( !v11 || !v3 )
      return v11;
    v13 = sub_E0E790((__int64)(a1 + 102), 24, v7, v8, v9, v10);
    if ( v13 )
    {
      v14 = *(_WORD *)(v13 + 8);
      v15 = *(_WORD *)(v11 + 9);
      *(_QWORD *)(v13 + 16) = v11;
      v11 = v13;
      *(_DWORD *)(v13 + 12) = v3;
      *(_WORD *)(v13 + 8) = v14 & 0xC000 | 3;
      *(_WORD *)(v13 + 9) = v15 & 0xFC0 | *(_WORD *)(v13 + 9) & 0xF03F;
      *(_QWORD *)v13 = &unk_49DEE88;
      return v11;
    }
    return 0;
  }
  *a1 = v2 + 1;
  v16 = sub_E0F8D0((__int64)a1);
  v19 = v16;
  v20 = v17;
  if ( !v16 )
    return 0;
  v21 = a1[1];
  v22 = (_BYTE *)*a1;
  if ( v16 > 8 && *(_QWORD *)v17 == 0x746F7270636A626FLL && *(_BYTE *)(v17 + 8) == 111 )
  {
    *a1 = v17 + 9;
    a1[1] = v16 + v17;
    v30 = sub_E0F8D0((__int64)a1);
    a1[1] = v31;
    *a1 = v32;
    v33 = v30;
    v35 = v34;
    if ( !v30 )
      return 0;
    v40 = sub_E20040(a1);
    if ( !v40 )
      return 0;
    v41 = sub_E0E790((__int64)(a1 + 102), 40, v36, v37, v38, v39);
    v11 = v41;
    if ( v41 )
    {
      *(_WORD *)(v41 + 8) = 16395;
      v42 = *(_BYTE *)(v41 + 10);
      *(_QWORD *)(v11 + 16) = v40;
      *(_QWORD *)(v11 + 24) = v33;
      *(_QWORD *)(v11 + 32) = v35;
      *(_BYTE *)(v11 + 10) = v42 & 0xF0 | 5;
      *(_QWORD *)v11 = &unk_49DF1E8;
    }
  }
  else
  {
    v43 = 0;
    if ( (_BYTE *)v21 != v22 && *v22 == 73 )
    {
      v43 = sub_E1F700((__int64)a1, 0, v17, v18, (__int64)v22, v21);
      if ( !v43 )
        return 0;
    }
    v27 = sub_E20040(a1);
    if ( !v27 )
      return 0;
    v28 = sub_E0E790((__int64)(a1 + 102), 48, v23, v24, v25, v26);
    v11 = v28;
    if ( v28 )
    {
      *(_QWORD *)(v28 + 16) = v27;
      *(_WORD *)(v28 + 8) = 16386;
      v29 = *(_BYTE *)(v28 + 10);
      *(_QWORD *)(v11 + 24) = v19;
      *(_QWORD *)(v11 + 32) = v20;
      *(_BYTE *)(v11 + 10) = v29 & 0xF0 | 5;
      *(_QWORD *)v11 = &unk_49DEE28;
      *(_QWORD *)(v11 + 40) = v43;
    }
  }
  return v11;
}
