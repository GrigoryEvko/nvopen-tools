// Function: sub_20EAFC0
// Address: 0x20eafc0
//
_QWORD *__fastcall sub_20EAFC0(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 (*v28)(void); // rsi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 (*v31)(); // rax
  __int64 *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 *v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 *v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 (*v57)(void); // rdx
  __int64 v58; // rax
  __int64 (*v59)(void); // rdx
  __int64 v60; // rax
  __int64 *v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rcx
  int v68; // r8d
  int v69; // r9d

  v5 = (_QWORD *)sub_22077B0(824);
  v6 = v5;
  if ( v5 )
  {
    v7 = *(__int64 **)(a1 + 8);
    v5[1] = a2;
    *v5 = off_4A00A30;
    v8 = *v7;
    v9 = v7[1];
    if ( v8 == v9 )
LABEL_85:
      BUG();
    while ( *(_UNKNOWN **)v8 != &unk_4FC450C )
    {
      v8 += 16;
      if ( v9 == v8 )
        goto LABEL_85;
    }
    v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4FC450C);
    v11 = *(__int64 **)(a1 + 8);
    v6[2] = v10;
    v12 = *v11;
    v13 = v11[1];
    if ( *v11 == v13 )
LABEL_86:
      BUG();
    while ( *(_UNKNOWN **)v12 != &unk_4FC452C )
    {
      v12 += 16;
      if ( v13 == v12 )
        goto LABEL_86;
    }
    v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
            *(_QWORD *)(v12 + 8),
            &unk_4FC452C);
    v15 = *(__int64 **)(a1 + 8);
    v6[3] = v14;
    v16 = *v15;
    v17 = v15[1];
    if ( v16 == v17 )
LABEL_87:
      BUG();
    while ( *(_UNKNOWN **)v16 != &unk_4F96DB4 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_87;
    }
    v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
            *(_QWORD *)(v16 + 8),
            &unk_4F96DB4);
    v19 = *(__int64 **)(a1 + 8);
    v6[4] = *(_QWORD *)(v18 + 160);
    v20 = *v19;
    v21 = v19[1];
    if ( v20 == v21 )
LABEL_76:
      BUG();
    while ( *(_UNKNOWN **)v20 != &unk_4FC62EC )
    {
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_76;
    }
    v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
            *(_QWORD *)(v20 + 8),
            &unk_4FC62EC);
    v23 = *(__int64 **)(a1 + 8);
    v6[5] = v22;
    v24 = *v23;
    v25 = v23[1];
    if ( v24 == v25 )
LABEL_81:
      BUG();
    while ( *(_UNKNOWN **)v24 != &unk_4FC6A0C )
    {
      v24 += 16;
      if ( v25 == v24 )
        goto LABEL_81;
    }
    v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(
            *(_QWORD *)(v24 + 8),
            &unk_4FC6A0C);
    v27 = a2[2];
    v6[6] = v26;
    v6[7] = a3;
    v6[8] = a2[5];
    v28 = *(__int64 (**)(void))(*(_QWORD *)v27 + 40LL);
    v29 = 0;
    if ( v28 != sub_1D00B00 )
    {
      v29 = v28();
      v27 = a2[2];
    }
    v6[9] = v29;
    v30 = 0;
    v31 = *(__int64 (**)())(*(_QWORD *)v27 + 112LL);
    if ( v31 != sub_1D00B10 )
      v30 = ((__int64 (__fastcall *)(__int64, _QWORD))v31)(v27, 0);
    v6[10] = v30;
    v32 = *(__int64 **)(a1 + 8);
    v33 = *v32;
    v34 = v32[1];
    if ( v33 == v34 )
LABEL_84:
      BUG();
    while ( *(_UNKNOWN **)v33 != &unk_4FC453D )
    {
      v33 += 16;
      if ( v34 == v33 )
        goto LABEL_84;
    }
    v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(
            *(_QWORD *)(v33 + 8),
            &unk_4FC453D);
    v6[21] = 0;
    v6[22] = v6 + 26;
    v6[23] = v6 + 26;
    v6[11] = v35;
    v6[35] = v6 + 39;
    v6[36] = v6 + 39;
    v6[15] = v6 + 17;
    v6[47] = v6 + 49;
    v36 = *(__int64 **)(a1 + 8);
    v6[16] = 0x800000000LL;
    v6[48] = 0x800000000LL;
    v6[24] = 8;
    *((_DWORD *)v6 + 50) = 0;
    v6[34] = 0;
    v6[37] = 8;
    *((_DWORD *)v6 + 76) = 0;
    v6[57] = off_4A00A68;
    v6[58] = a2;
    v37 = *v36;
    v38 = v36[1];
    if ( v37 == v38 )
LABEL_82:
      BUG();
    while ( *(_UNKNOWN **)v37 != &unk_4FC450C )
    {
      v37 += 16;
      if ( v38 == v37 )
        goto LABEL_82;
    }
    v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v37 + 8) + 104LL))(
            *(_QWORD *)(v37 + 8),
            &unk_4FC450C);
    v40 = *(__int64 **)(a1 + 8);
    v6[59] = v39;
    v41 = *v40;
    v42 = v40[1];
    if ( v41 == v42 )
LABEL_83:
      BUG();
    while ( *(_UNKNOWN **)v41 != &unk_4FC452C )
    {
      v41 += 16;
      if ( v42 == v41 )
        goto LABEL_83;
    }
    v43 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v41 + 8) + 104LL))(
            *(_QWORD *)(v41 + 8),
            &unk_4FC452C);
    v44 = *(__int64 **)(a1 + 8);
    v6[60] = v43;
    v45 = *v44;
    v46 = v44[1];
    if ( v45 == v46 )
LABEL_77:
      BUG();
    while ( *(_UNKNOWN **)v45 != &unk_4F96DB4 )
    {
      v45 += 16;
      if ( v46 == v45 )
        goto LABEL_77;
    }
    v47 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v45 + 8) + 104LL))(
            *(_QWORD *)(v45 + 8),
            &unk_4F96DB4);
    v48 = *(__int64 **)(a1 + 8);
    v6[61] = *(_QWORD *)(v47 + 160);
    v49 = *v48;
    v50 = v48[1];
    if ( v49 == v50 )
LABEL_78:
      BUG();
    while ( *(_UNKNOWN **)v49 != &unk_4FC62EC )
    {
      v49 += 16;
      if ( v50 == v49 )
        goto LABEL_78;
    }
    v51 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v49 + 8) + 104LL))(
            *(_QWORD *)(v49 + 8),
            &unk_4FC62EC);
    v52 = *(__int64 **)(a1 + 8);
    v6[62] = v51;
    v53 = *v52;
    v54 = v52[1];
    if ( v53 == v54 )
LABEL_80:
      BUG();
    while ( *(_UNKNOWN **)v53 != &unk_4FC6A0C )
    {
      v53 += 16;
      if ( v54 == v53 )
        goto LABEL_80;
    }
    v55 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v53 + 8) + 104LL))(
            *(_QWORD *)(v53 + 8),
            &unk_4FC6A0C);
    v56 = a2[2];
    v6[63] = v55;
    v6[64] = a3;
    v6[65] = a2[5];
    v57 = *(__int64 (**)(void))(*(_QWORD *)v56 + 40LL);
    v58 = 0;
    if ( v57 != sub_1D00B00 )
    {
      v58 = v57();
      v56 = a2[2];
    }
    v6[66] = v58;
    v59 = *(__int64 (**)(void))(*(_QWORD *)v56 + 112LL);
    v60 = 0;
    if ( v59 != sub_1D00B10 )
      v60 = v59();
    v61 = *(__int64 **)(a1 + 8);
    v6[67] = v60;
    v62 = *v61;
    v63 = v61[1];
    if ( v62 == v63 )
LABEL_79:
      BUG();
    while ( *(_UNKNOWN **)v62 != &unk_4FC453D )
    {
      v62 += 16;
      if ( v63 == v62 )
        goto LABEL_79;
    }
    v64 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v62 + 8) + 104LL))(
            *(_QWORD *)(v62 + 8),
            &unk_4FC453D);
    v65 = v6[59];
    v66 = (__int64)(a2[13] - a2[12]) >> 3;
    v6[68] = v64;
    sub_1F139C0(v6 + 69, v65, v66, v67, v68, v69);
    v6[88] = 0;
    v6[89] = 0;
    v6[90] = 0;
    *((_DWORD *)v6 + 182) = 0;
    v6[92] = 0;
    v6[93] = 0;
    v6[94] = 0;
    *((_DWORD *)v6 + 190) = 0;
    v6[96] = 0;
    v6[97] = 0;
    v6[98] = 0;
    v6[99] = 0;
    v6[100] = 0;
    v6[101] = 0;
    *((_DWORD *)v6 + 204) = 0;
  }
  return v6;
}
