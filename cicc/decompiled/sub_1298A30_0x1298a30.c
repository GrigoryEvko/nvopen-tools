// Function: sub_1298A30
// Address: 0x1298a30
//
__int64 __fastcall sub_1298A30(__int64 a1, __int64 a2, _DWORD *a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // rdi
  unsigned __int64 *v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 result; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned int v20; // r8d
  __int64 v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // r13
  __int64 v24; // rdi
  unsigned __int64 *v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // r15
  __int64 v31; // rax
  bool v32; // al
  unsigned int v33; // [rsp+Ch] [rbp-64h]
  __int64 v34; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v35[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v36; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 136);
  if ( !v4 )
    goto LABEL_5;
  v5 = *(_QWORD *)(a2 + 16);
  v7 = *(_QWORD *)(v5 + 24);
  v8 = *(_DWORD *)(v5 + 12);
  if ( v8 == 2 )
  {
    if ( !sub_127B420(v7) )
      sub_127B550("Indirect returns for non-aggregate values not supported!", a3, 1);
    goto LABEL_5;
  }
  if ( v8 > 2 )
  {
    if ( v8 != 3 )
      sub_127B550("Unsupported ABI variant!", a3, 1);
    goto LABEL_5;
  }
  v19 = *(_QWORD *)(a1 + 32);
  v36 = 257;
  v20 = unk_4D0463C;
  if ( unk_4D0463C )
  {
    v32 = sub_126A420(v19, v4);
    v4 = *(_QWORD *)(a1 + 136);
    v19 = *(_QWORD *)(a1 + 32);
    v20 = v32;
  }
  v33 = v20;
  v21 = sub_127A030(v19 + 8, v7, 0);
  v22 = sub_1648A60(64, 1);
  v23 = (_QWORD *)v22;
  if ( v22 )
    sub_15F9210(v22, v21, v4, 0, v33, 0);
  v24 = *(_QWORD *)(a1 + 56);
  if ( v24 )
  {
    v25 = *(unsigned __int64 **)(a1 + 64);
    sub_157E9D0(v24 + 40, v23);
    v26 = v23[3];
    v27 = *v25;
    v23[4] = v25;
    v27 &= 0xFFFFFFFFFFFFFFF8LL;
    v23[3] = v27 | v26 & 7;
    *(_QWORD *)(v27 + 8) = v23 + 3;
    *v25 = *v25 & 7 | (unsigned __int64)(v23 + 3);
  }
  sub_164B780(v23, v35);
  v28 = *(_QWORD *)(a1 + 48);
  if ( v28 )
  {
    v34 = *(_QWORD *)(a1 + 48);
    sub_1623A60(&v34, v28, 2);
    if ( v23[6] )
      sub_161E7C0(v23 + 6);
    v29 = v34;
    v23[6] = v34;
    if ( v29 )
      sub_1623210(&v34, v29, v23 + 6);
    sub_15F8F50(v23, *(unsigned int *)(a1 + 144));
    goto LABEL_27;
  }
  sub_15F8F50(v23, *(unsigned int *)(a1 + 144));
  if ( !v23 )
  {
LABEL_5:
    v9 = *(_QWORD *)(a1 + 72);
    v36 = 257;
    v10 = sub_1648A60(56, 0);
    v11 = (_QWORD *)v10;
    if ( v10 )
      sub_15F6F90(v10, v9, 0, 0);
    goto LABEL_7;
  }
LABEL_27:
  v30 = *(_QWORD *)(a1 + 72);
  v36 = 257;
  v31 = sub_1648A60(56, 1);
  v11 = (_QWORD *)v31;
  if ( v31 )
    sub_15F6F90(v31, v30, v23, 0);
LABEL_7:
  v12 = *(_QWORD *)(a1 + 56);
  if ( v12 )
  {
    v13 = *(unsigned __int64 **)(a1 + 64);
    sub_157E9D0(v12 + 40, v11);
    v14 = v11[3];
    v15 = *v13;
    v11[4] = v13;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v15 | v14 & 7;
    *(_QWORD *)(v15 + 8) = v11 + 3;
    *v13 = *v13 & 7 | (unsigned __int64)(v11 + 3);
  }
  result = sub_164B780(v11, v35);
  v17 = *(_QWORD *)(a1 + 48);
  if ( v17 )
  {
    v34 = *(_QWORD *)(a1 + 48);
    result = sub_1623A60(&v34, v17, 2);
    if ( v11[6] )
      result = sub_161E7C0(v11 + 6);
    v18 = v34;
    v11[6] = v34;
    if ( v18 )
      return sub_1623210(&v34, v18, v11 + 6);
  }
  return result;
}
