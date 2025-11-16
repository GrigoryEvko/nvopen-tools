// Function: sub_2AD6730
// Address: 0x2ad6730
//
__int64 __fastcall sub_2AD6730(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 v20; // r15
  int v21; // eax
  char v22; // bl
  __int64 (__fastcall *v23)(__int64); // rax
  char *v24; // rax
  __int64 v25; // rcx
  const char *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v34; // r13
  __int64 *v35; // rbx
  void **v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // r9
  __int64 v40; // rbx
  __int64 v41; // rsi
  _QWORD *v42; // r9
  unsigned __int64 v43; // rcx
  _QWORD *v44; // r9
  __int64 v45; // r9
  __int64 v46; // r15
  __int64 *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rbx
  bool v51; // zf
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 *v56; // rax
  void **v57; // rax
  int v58; // esi
  int v59; // edx
  unsigned int v60; // esi
  _QWORD *v61; // [rsp+0h] [rbp-160h]
  _QWORD *v62; // [rsp+0h] [rbp-160h]
  _QWORD *v63; // [rsp+0h] [rbp-160h]
  __int64 v64; // [rsp+8h] [rbp-158h]
  int v65; // [rsp+10h] [rbp-150h]
  void *v66; // [rsp+10h] [rbp-150h]
  __int64 v67; // [rsp+18h] [rbp-148h]
  __int64 v69; // [rsp+28h] [rbp-138h]
  __int64 v70; // [rsp+28h] [rbp-138h]
  _QWORD *v71; // [rsp+28h] [rbp-138h]
  __int64 *v72; // [rsp+30h] [rbp-130h]
  void *v73; // [rsp+38h] [rbp-128h]
  _QWORD *v74; // [rsp+38h] [rbp-128h]
  __int64 i; // [rsp+40h] [rbp-120h]
  __int64 v77; // [rsp+58h] [rbp-108h]
  void *v78; // [rsp+68h] [rbp-F8h] BYREF
  void *v79; // [rsp+70h] [rbp-F0h] BYREF
  void *v80; // [rsp+78h] [rbp-E8h] BYREF
  void *v81; // [rsp+80h] [rbp-E0h] BYREF
  void *v82; // [rsp+88h] [rbp-D8h] BYREF
  _QWORD v83[2]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v84; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 *v85; // [rsp+A8h] [rbp-B8h]
  __int64 *v86; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-A8h]
  __int64 v88; // [rsp+C0h] [rbp-A0h]
  void *v89[4]; // [rsp+D0h] [rbp-90h] BYREF
  __int16 v90; // [rsp+F0h] [rbp-70h]
  __int64 v91; // [rsp+100h] [rbp-60h] BYREF
  __int64 v92; // [rsp+108h] [rbp-58h]
  __int64 v93; // [rsp+110h] [rbp-50h]
  __int64 v94; // [rsp+118h] [rbp-48h]
  _QWORD *v95; // [rsp+120h] [rbp-40h]
  __int64 v96; // [rsp+128h] [rbp-38h]

  v3 = sub_2BF3F10(a2);
  v4 = sub_2BF04D0(v3);
  if ( v4 + 112 == (*(_QWORD *)(v4 + 112) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( *(_DWORD *)(v4 + 88) != 1 )
      BUG();
    v4 = **(_QWORD **)(v4 + 80);
  }
  v5 = *(_QWORD *)(v4 + 120);
  if ( !v5 )
    BUG();
  if ( !*(_DWORD *)(v5 + 32) )
    BUG();
  v6 = **(_QWORD **)(v5 + 24);
  v91 = 0;
  v7 = *(_QWORD **)(*(_QWORD *)(v6 + 40) + 8LL);
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = v7;
  v96 = *v7;
  v8 = *(_QWORD *)(a2 + 8);
  if ( *(_DWORD *)(v8 + 64) != 1 )
    BUG();
  v9 = *(__int64 **)(v8 + 56);
  v10 = 0;
  v11 = *v9;
  if ( *(_DWORD *)(*v9 + 64) == 1 )
    v10 = **(_QWORD **)(v11 + 56);
  v12 = sub_2BF3F10(a2);
  v77 = 0;
  if ( *(_DWORD *)(v12 + 64) == 1 )
    v77 = **(_QWORD **)(v12 + 56);
  v84 = v11;
  v83[0] = v10;
  v83[1] = sub_2BF05A0(v10);
  v85 = (__int64 *)(v11 + 112);
  v13 = sub_2BF3F10(a2);
  v14 = sub_2BF04D0(v13);
  if ( v14 + 112 == (*(_QWORD *)(v14 + 112) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( *(_DWORD *)(v14 + 88) != 1 )
      BUG();
    v14 = **(_QWORD **)(v14 + 80);
  }
  v15 = *(_QWORD *)(v14 + 120);
  if ( !v15 )
    BUG();
  if ( !*(_DWORD *)(v15 + 32) )
    BUG();
  v16 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v15 + 24) + 40LL) + 8LL), 1, 0);
  v64 = sub_2AC42A0(a2, v16);
  v17 = *(_QWORD *)(a2 + 8);
  v18 = *(_QWORD *)(v17 + 120);
  v67 = v77 + 112;
  for ( i = v17 + 112; i != v18; v18 = *(_QWORD *)(v18 + 8) )
  {
    if ( !v18 )
      BUG();
    if ( **(_BYTE **)(v18 + 72) != 84 )
      return sub_C7D6A0(v92, 16LL * (unsigned int)v94, 8);
    v89[0] = *(void **)(v18 + 72);
    v19 = sub_2AC6480(a1 + 128, (__int64 *)v89);
    v20 = *v19;
    v21 = *(unsigned __int8 *)(*v19 + 8);
    v22 = v21;
    if ( (_BYTE)v21 == 33 )
    {
      v35 = (__int64 *)(a2 + 216);
      v72 = *(__int64 **)(v20 + 160);
      if ( v72 )
        continue;
      v57 = *(void ***)(v20 + 48);
      v34 = *(_QWORD *)(v20 + 152);
      v73 = v57[1];
      if ( *(_DWORD *)(v20 + 56) )
        v72 = (__int64 *)*v57;
      if ( (unsigned __int8)sub_2C1B260(v20) )
        goto LABEL_35;
    }
    else
    {
      if ( v21 != 34 )
      {
        v23 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v20 + 40LL);
        if ( v23 == sub_2AA7530 )
          v24 = *(char **)(*(_QWORD *)(v20 + 48) + 8LL);
        else
          v24 = (char *)v23(v20);
        v25 = 12;
        v26 = "bc.merge.rdx";
        if ( v22 == 32 )
        {
          v86 = (__int64 *)v24;
          v89[0] = "vector.recur.extract";
          v90 = 259;
          v87 = v64;
          v24 = (char *)sub_2AB07D0(v83, 82, (__int64 *)&v86, 2, 0, v89);
          v26 = "scalar.recur.init";
          if ( v24 )
            v24 += 96;
          v25 = 17;
        }
        v86 = (__int64 *)v24;
        v27 = 0;
        v90 = 261;
        v89[0] = (void *)v26;
        v89[1] = (void *)v25;
        if ( *(_DWORD *)(v20 + 56) )
          v27 = **(_QWORD **)(v20 + 48);
        v87 = v27;
        v30 = sub_2AB07D0(&v84, 75, (__int64 *)&v86, 2, 0, v89);
        if ( v30 )
          v30 += 96;
        sub_2AAECA0(v18 + 16, v30, v28, v29, v31, v32);
        continue;
      }
      v34 = *(_QWORD *)(v20 + 152);
      v35 = (__int64 *)(a2 + 216);
      v36 = *(void ***)(v20 + 48);
      v73 = v36[1];
      if ( *(_DWORD *)(v20 + 56) )
        v72 = (__int64 *)*v36;
      else
        v72 = 0;
    }
    v90 = 257;
    v37 = *(_QWORD *)(v34 + 40);
    v69 = v37;
    if ( !v37 || !(unsigned __int8)sub_920620(v37) )
      v69 = 0;
    v65 = *(_DWORD *)(v34 + 24);
    v38 = (_QWORD *)sub_22077B0(0xC8u);
    if ( v38 )
    {
      v87 = (__int64)v35;
      v40 = (__int64)v73;
      v86 = v72;
      v74 = v38;
      v82 = 0;
      v88 = v40;
      sub_2AAF4A0((__int64)v38, 1, (__int64 *)&v86, 3, (__int64 *)&v82, v39);
      sub_9C6650(&v82);
      *v74 = &unk_4A23718;
      v74[5] = &unk_4A23758;
      v74[12] = &unk_4A23790;
      *((_DWORD *)v74 + 38) = v65;
      v74[20] = v69;
      sub_CA0F50(v74 + 21, v89);
      v38 = v74;
      if ( v77 )
      {
LABEL_33:
        v38[10] = v77;
        v41 = *(_QWORD *)(v77 + 112);
        v38[4] = v67;
        v41 &= 0xFFFFFFFFFFFFFFF8LL;
        v38[3] = v41 | v38[3] & 7LL;
        *(_QWORD *)(v41 + 8) = v38 + 3;
        *(_QWORD *)(v77 + 112) = *(_QWORD *)(v77 + 112) & 7LL | (unsigned __int64)(v38 + 3);
      }
      v35 = v38 + 12;
      goto LABEL_35;
    }
    if ( v77 )
      goto LABEL_33;
    v35 = 0;
LABEL_35:
    v66 = (void *)(v20 + 96);
    v70 = sub_2BFD6A0(&v91, v20 + 96);
    if ( v70 == sub_2BFD6A0(&v91, v35) )
      goto LABEL_47;
    v81 = *(void **)(v20 + 88);
    if ( v81 )
    {
      sub_2AAAFA0((__int64 *)&v81);
      v82 = v81;
      if ( v81 )
        sub_2AAAFA0((__int64 *)&v82);
    }
    else
    {
      v82 = 0;
    }
    v42 = (_QWORD *)sub_22077B0(0xA8u);
    if ( v42 )
    {
      v89[0] = v82;
      if ( v82 )
      {
        v61 = v42;
        sub_2AAAFA0((__int64 *)v89);
        v42 = v61;
      }
      v62 = v42;
      v86 = v35;
      sub_2AAF4A0((__int64)v42, 10, (__int64 *)&v86, 1, (__int64 *)v89, (__int64)v42);
      sub_9C6650(v89);
      v42 = v62;
      *((_DWORD *)v62 + 38) = 38;
      *v62 = &unk_4A24560;
      v62[12] = &unk_4A245D8;
      v62[5] = &unk_4A245A0;
      v62[20] = v70;
    }
    if ( v77 )
    {
      v42[10] = v77;
      v63 = v42;
      v43 = *(_QWORD *)(v77 + 112) & 0xFFFFFFFFFFFFFFF8LL;
      v42[4] = v67;
      v42[3] = v43 | v42[3] & 7LL;
      *(_QWORD *)(v43 + 8) = v42 + 3;
      *(_QWORD *)(v77 + 112) = *(_QWORD *)(v77 + 112) & 7LL | (unsigned __int64)(v42 + 3);
      sub_9C6650(&v82);
      v44 = v63;
    }
    else
    {
      v71 = v42;
      v35 = 0;
      sub_9C6650(&v82);
      v44 = v71;
      if ( !v71 )
        goto LABEL_46;
    }
    v35 = v44 + 12;
LABEL_46:
    sub_9C6650(&v81);
LABEL_47:
    v89[0] = "bc.resume.val";
    v90 = 259;
    v78 = *(void **)(v20 + 88);
    if ( !v78 )
    {
      v86 = v35;
      v79 = 0;
      v87 = (__int64)v72;
LABEL_73:
      v80 = 0;
      goto LABEL_51;
    }
    sub_2AAAFA0((__int64 *)&v78);
    v86 = v35;
    v87 = (__int64)v72;
    v79 = v78;
    if ( !v78 )
      goto LABEL_73;
    sub_2AAAFA0((__int64 *)&v79);
    v80 = v79;
    if ( v79 )
      sub_2AAAFA0((__int64 *)&v80);
LABEL_51:
    v46 = sub_22077B0(0xC8u);
    if ( v46 )
    {
      v81 = v80;
      if ( v80 )
      {
        sub_2AAAFA0((__int64 *)&v81);
        v82 = v81;
        if ( v81 )
          sub_2AAAFA0((__int64 *)&v82);
      }
      else
      {
        v82 = 0;
      }
      sub_2AAF4A0(v46, 4, (__int64 *)&v86, 2, (__int64 *)&v82, v45);
      sub_9C6650(&v82);
      *(_BYTE *)(v46 + 152) = 7;
      *(_DWORD *)(v46 + 156) = 0;
      *(_QWORD *)v46 = &unk_4A23258;
      *(_QWORD *)(v46 + 40) = &unk_4A23290;
      *(_QWORD *)(v46 + 96) = &unk_4A232C8;
      sub_9C6650(&v81);
      *(_BYTE *)(v46 + 160) = 75;
      *(_QWORD *)v46 = &unk_4A23B70;
      *(_QWORD *)(v46 + 40) = &unk_4A23BB8;
      *(_QWORD *)(v46 + 96) = &unk_4A23BF0;
      sub_CA0F50((__int64 *)(v46 + 168), v89);
    }
    if ( v84 )
    {
      v47 = v85;
      *(_QWORD *)(v46 + 80) = v84;
      v48 = *(_QWORD *)(v46 + 24);
      v49 = *v47;
      *(_QWORD *)(v46 + 32) = v47;
      v49 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v46 + 24) = v49 | v48 & 7;
      *(_QWORD *)(v49 + 8) = v46 + 24;
      *v47 = *v47 & 7 | (v46 + 24);
      sub_9C6650(&v80);
      sub_9C6650(&v79);
      sub_9C6650(&v78);
    }
    else
    {
      sub_9C6650(&v80);
      sub_9C6650(&v79);
      sub_9C6650(&v78);
      if ( !v46 )
        continue;
    }
    v50 = **(_QWORD **)(v46 + 48);
    v82 = v66;
    v51 = (unsigned __int8)sub_2AC2170(a3, (__int64 *)&v82, &v86) == 0;
    v56 = v86;
    if ( !v51 )
      goto LABEL_59;
    v89[0] = v86;
    v58 = *(_DWORD *)(a3 + 16);
    ++*(_QWORD *)a3;
    v59 = v58 + 1;
    v60 = *(_DWORD *)(a3 + 24);
    v54 = 2 * v60;
    if ( 4 * v59 >= 3 * v60 )
    {
      v60 *= 2;
LABEL_89:
      sub_2AD6580(a3, v60);
      sub_2AC2170(a3, (__int64 *)&v82, v89);
      v59 = *(_DWORD *)(a3 + 16) + 1;
      v56 = (__int64 *)v89[0];
      goto LABEL_77;
    }
    if ( v60 - *(_DWORD *)(a3 + 20) - v59 <= v60 >> 3 )
      goto LABEL_89;
LABEL_77:
    v53 = a3;
    *(_DWORD *)(a3 + 16) = v59;
    if ( *v56 != -4096 )
      --*(_DWORD *)(a3 + 20);
    v52 = (__int64)v82;
    v56[1] = 0;
    *v56 = v52;
LABEL_59:
    v56[1] = v50;
    sub_2AAECA0(v18 + 16, v46 + 96, v52, v53, v54, v55);
  }
  return sub_C7D6A0(v92, 16LL * (unsigned int)v94, 8);
}
