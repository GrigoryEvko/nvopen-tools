// Function: sub_14809B0
// Address: 0x14809b0
//
__int64 __fastcall sub_14809B0(__int64 *a1, __int64 *a2, __int64 a3, unsigned int a4, __int64 a5)
{
  _WORD *v5; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // edx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int16 v21; // r8
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  __int16 v29; // r8
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // [rsp+Ch] [rbp-44h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+18h] [rbp-38h]
  __int64 v39; // [rsp+18h] [rbp-38h]

  v5 = (_WORD *)*a1;
  if ( *(_WORD *)(*a1 + 24) != 7 )
    return 0;
  if ( *((_QWORD *)v5 + 6) != a3 )
    return 0;
  if ( *((_QWORD *)v5 + 5) != 2 )
    return 0;
  v9 = sub_13A5BC0((_QWORD *)*a1, a5);
  if ( *(_WORD *)(v9 + 24) )
    return 0;
  v10 = sub_14806B0(a5, *a2, **((_QWORD **)v5 + 4), 0, 0);
  if ( *(_WORD *)(v10 + 24) )
    return 0;
  v11 = *(_QWORD *)(v10 + 32);
  v12 = *(_DWORD *)(v11 + 32);
  v13 = *(_QWORD *)(v11 + 24);
  if ( (_BYTE)a4 )
  {
    if ( v12 > 0x40 )
      v14 = *(_QWORD *)v13;
    else
      v14 = (__int64)(v13 << (64 - (unsigned __int8)v12)) >> (64 - (unsigned __int8)v12);
    v34 = v14;
    v15 = *(_QWORD *)(v9 + 32);
    v16 = *(_DWORD *)(v15 + 32);
    v17 = *(__int64 **)(v15 + 24);
    if ( v16 > 0x40 )
      v18 = *v17;
    else
      v18 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
    v33 = a4;
    v38 = v18;
    v19 = sub_1456040(v9);
    v20 = sub_145CF80(a5, v19, v38 * (v34 / v38 + 1), 0);
    v21 = v5[13];
    v22 = *((_QWORD *)v5 + 6);
    v35 = v20;
    LODWORD(v38) = v21 & 7;
    v23 = sub_1456040(v9);
    v24 = sub_145CF80(a5, v23, 0, 0);
    *a1 = sub_14799E0(a5, v24, v9, v22, v38);
    *a2 = v35;
    return v33;
  }
  else
  {
    if ( v12 > 0x40 )
      v13 = *(_QWORD *)v13;
    v25 = *(_QWORD *)(v9 + 32);
    v26 = *(_QWORD *)(v25 + 24);
    if ( *(_DWORD *)(v25 + 32) > 0x40u )
      v26 = *(_QWORD *)v26;
    v39 = v26 * (v13 / v26 + 1);
    v27 = sub_1456040(v9);
    v28 = sub_145CF80(a5, v27, v39, 0);
    v29 = v5[13];
    v30 = *((_QWORD *)v5 + 6);
    v36 = v28;
    LODWORD(v39) = v29 & 7;
    v31 = sub_1456040(v9);
    v32 = sub_145CF80(a5, v31, 0, 0);
    *a1 = sub_14799E0(a5, v32, v9, v30, v39);
    *a2 = v36;
    return 1;
  }
}
