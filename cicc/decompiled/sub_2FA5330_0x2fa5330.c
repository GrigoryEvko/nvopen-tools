// Function: sub_2FA5330
// Address: 0x2fa5330
//
__int64 __fastcall sub_2FA5330(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v41; // rdi
  __int64 (*v42)(); // rax
  __int64 v43; // rdi
  __int64 (*v44)(); // rax

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
    goto LABEL_41;
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_41;
  }
  v7 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_5027190)
                 + 256);
  *(_QWORD *)(a1 + 176) = v7;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 16LL);
  if ( v8 == sub_23CE270 )
  {
    *(_QWORD *)(a1 + 184) = 0;
    goto LABEL_41;
  }
  v9 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v7, a2);
  *(_QWORD *)(a1 + 184) = v9;
  v10 = v9;
  v11 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
  if ( v11 == sub_2C8F680 )
  {
    *(_QWORD *)(a1 + 192) = 0;
    BUG();
  }
  v12 = ((__int64 (__fastcall *)(__int64))v11)(v10);
  *(_QWORD *)(a1 + 192) = v12;
  v13 = v12;
  v14 = *(__int64 (**)())(*(_QWORD *)v12 + 104LL);
  if ( v14 != sub_2D56590 && !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v14)(v13, 0) )
  {
    v41 = *(_QWORD *)(a1 + 192);
    v42 = *(__int64 (**)())(*(_QWORD *)v41 + 104LL);
    if ( v42 != sub_2D56590 && !((unsigned __int8 (__fastcall *)(__int64, __int64))v42)(v41, 1) )
    {
      v43 = *(_QWORD *)(a1 + 192);
      v44 = *(__int64 (**)())(*(_QWORD *)v43 + 104LL);
      if ( v44 != sub_2D56590 && !((unsigned __int8 (__fastcall *)(__int64, __int64))v44)(v43, 2) )
        return 0;
    }
  }
  v15 = *(__int64 **)(a1 + 8);
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
    goto LABEL_41;
  while ( *(_UNKNOWN **)v16 != &unk_4F89C28 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_41;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F89C28);
  v19 = sub_DFED00(v18, a2);
  *(_QWORD *)(a1 + 200) = v19;
  if ( !(unsigned __int8)sub_DFAC40(v19) )
    return 0;
  v20 = *(__int64 **)(a1 + 8);
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
    goto LABEL_41;
  while ( *(_UNKNOWN **)v21 != &unk_4F875EC )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_41;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(*(_QWORD *)(v21 + 8), &unk_4F875EC);
  v24 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 208) = v23 + 176;
  v25 = *v24;
  v26 = v24[1];
  if ( v25 == v26 )
    goto LABEL_41;
  while ( *(_UNKNOWN **)v25 != &unk_4F8D9B0 )
  {
    v25 += 16;
    if ( v26 == v25 )
      goto LABEL_41;
  }
  v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(*(_QWORD *)(v25 + 8), &unk_4F8D9B0);
  v28 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 216) = v27 + 176;
  v29 = *v28;
  v30 = v28[1];
  if ( v29 == v30 )
    goto LABEL_41;
  while ( *(_UNKNOWN **)v29 != &unk_4F87C64 )
  {
    v29 += 16;
    if ( v30 == v29 )
      goto LABEL_41;
  }
  v31 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v29 + 8) + 104LL))(*(_QWORD *)(v29 + 8), &unk_4F87C64);
  v32 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 224) = *(_QWORD *)(v31 + 176);
  v33 = *v32;
  v34 = v32[1];
  if ( v33 == v34 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v33 != &unk_4F8FAE4 )
  {
    v33 += 16;
    if ( v34 == v33 )
      goto LABEL_41;
  }
  v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(*(_QWORD *)(v33 + 8), &unk_4F8FAE4);
  v36 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(v35 + 176);
  sub_2FF7BB0(a1 + 240, v36);
  if ( (unsigned __int8)sub_11F2A60(a2, *(_QWORD *)(a1 + 224), *(__int64 **)(a1 + 216)) )
    return 0;
  return sub_2FA48F0(a1 + 176, a2, v37, v38, v39);
}
