// Function: sub_2F736C0
// Address: 0x2f736c0
//
__int64 __fastcall sub_2F736C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // r13d
  bool v15; // zf
  _QWORD v17[8]; // [rsp+0h] [rbp-3E0h] BYREF
  _BYTE v18[320]; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 v19; // [rsp+180h] [rbp-260h]
  __int64 v20; // [rsp+188h] [rbp-258h]
  __int64 v21; // [rsp+190h] [rbp-250h]
  int v22; // [rsp+198h] [rbp-248h]
  __int64 v23; // [rsp+1A0h] [rbp-240h]
  __int64 v24; // [rsp+1A8h] [rbp-238h]
  __int64 v25; // [rsp+1B0h] [rbp-230h]
  int v26; // [rsp+1B8h] [rbp-228h]
  __int64 v27; // [rsp+1C0h] [rbp-220h]
  __int64 v28; // [rsp+1C8h] [rbp-218h]
  __int64 v29; // [rsp+1D0h] [rbp-210h]
  int v30; // [rsp+1D8h] [rbp-208h]
  __int128 v31; // [rsp+1E0h] [rbp-200h]
  __int16 v32; // [rsp+1F0h] [rbp-1F0h]
  char v33; // [rsp+1F2h] [rbp-1EEh]
  char *v34; // [rsp+1F8h] [rbp-1E8h]
  __int64 v35; // [rsp+200h] [rbp-1E0h]
  char v36; // [rsp+208h] [rbp-1D8h] BYREF
  char *v37; // [rsp+248h] [rbp-198h]
  __int64 v38; // [rsp+250h] [rbp-190h]
  char v39; // [rsp+258h] [rbp-188h] BYREF
  __int64 v40; // [rsp+298h] [rbp-148h]
  char *v41; // [rsp+2A0h] [rbp-140h]
  __int64 v42; // [rsp+2A8h] [rbp-138h]
  int v43; // [rsp+2B0h] [rbp-130h]
  char v44; // [rsp+2B4h] [rbp-12Ch]
  char v45; // [rsp+2B8h] [rbp-128h] BYREF
  char *v46; // [rsp+2F8h] [rbp-E8h]
  __int64 v47; // [rsp+300h] [rbp-E0h]
  char v48; // [rsp+308h] [rbp-D8h] BYREF
  char *v49; // [rsp+348h] [rbp-98h]
  __int64 v50; // [rsp+350h] [rbp-90h]
  char v51; // [rsp+358h] [rbp-88h] BYREF
  __int64 v52; // [rsp+378h] [rbp-68h]
  __int64 v53; // [rsp+380h] [rbp-60h]
  __int64 v54; // [rsp+388h] [rbp-58h]
  __int64 v55; // [rsp+390h] [rbp-50h]
  __int64 v56; // [rsp+398h] [rbp-48h]
  __int64 v57; // [rsp+3A0h] [rbp-40h]
  __int64 v58; // [rsp+3A8h] [rbp-38h]
  int v59; // [rsp+3B0h] [rbp-30h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501EACC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_20;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501EACC);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 200;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_50208AC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_19;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_50208AC)
      + 200;
  v11 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5025C1C);
  if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_5025C1C)) != 0 )
    v13 = v12 + 200;
  else
    v13 = 0;
  v17[6] = v13;
  v14 = 0;
  v17[7] = v10;
  v17[0] = off_4A2B718;
  memset(&v17[1], 0, 32);
  v17[5] = v7;
  sub_2F5FEE0((__int64)v18);
  v37 = &v39;
  v32 = 0;
  v41 = &v45;
  v34 = &v36;
  v46 = &v48;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v33 = 0;
  v35 = 0x800000000LL;
  v38 = 0x800000000LL;
  v40 = 0;
  v42 = 8;
  v43 = 0;
  v44 = 1;
  v47 = 0x800000000LL;
  v49 = &v51;
  v50 = 0x800000000LL;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v31 = 0;
  v56 = 0;
  v15 = *(_BYTE *)(a2 + 341) == 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  if ( v15 )
    v14 = sub_2F71140((__int64)v17, (_QWORD *)a2);
  sub_2F61430((__int64)v17);
  return v14;
}
