// Function: sub_27987E0
// Address: 0x27987e0
//
__int64 __fastcall sub_27987E0(__int64 a1, unsigned __int8 *a2)
{
  _QWORD *v2; // r12
  int v3; // ebx
  char v4; // cl
  unsigned int v5; // r15d
  unsigned __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // eax
  unsigned int *v15; // rax
  __int64 v16; // r8
  unsigned int v17; // ebx
  unsigned int v18; // edx
  int v19; // eax
  unsigned __int8 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int8 *v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int8 **v26; // rax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  const char *v33; // rax
  __int64 v34; // rsi
  int v35; // ebx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r15
  _QWORD *v39; // r14
  __int64 v40; // r12
  __int64 v41; // r15
  __int64 v42; // rbx
  __int64 v43; // rsi
  int v44; // eax
  int v45; // eax
  unsigned int v46; // edi
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rdi
  __int64 v50; // r14
  int v51; // eax
  int v52; // eax
  unsigned int v53; // ecx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rcx
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  _QWORD *v59; // rax
  __int64 v60; // rdx
  const char **v61; // r14
  const char *v62; // rsi
  __int64 v63; // rdi
  __int64 v64; // rcx
  int v65; // edx
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  unsigned __int64 v72; // rax
  __int64 v73; // r9
  __int64 v74; // rdx
  unsigned __int64 v75; // r13
  unsigned __int64 *v76; // rdx
  __int64 v77; // rax
  __int64 v78; // [rsp+8h] [rbp-158h]
  __int64 v79; // [rsp+10h] [rbp-150h]
  unsigned int v80; // [rsp+18h] [rbp-148h]
  unsigned int v81; // [rsp+1Ch] [rbp-144h]
  unsigned int v82; // [rsp+20h] [rbp-140h]
  unsigned int v83; // [rsp+20h] [rbp-140h]
  _QWORD *v84; // [rsp+20h] [rbp-140h]
  int v85; // [rsp+28h] [rbp-138h]
  __int64 v86; // [rsp+28h] [rbp-138h]
  __int64 v87; // [rsp+30h] [rbp-130h]
  __int64 v88; // [rsp+30h] [rbp-130h]
  __int64 v89; // [rsp+38h] [rbp-128h]
  __int64 v91; // [rsp+40h] [rbp-120h]
  _QWORD v92[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v93; // [rsp+60h] [rbp-100h]
  char *v94; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v95; // [rsp+78h] [rbp-E8h]
  const char *v96; // [rsp+80h] [rbp-E0h]
  __int16 v97; // [rsp+90h] [rbp-D0h]
  _BYTE *v98; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-B8h]
  _BYTE v100[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v2 = (_QWORD *)a1;
  v3 = *a2;
  v4 = v3 - 30;
  if ( (unsigned __int8)(v3 - 30) <= 0x36u )
  {
    v5 = ((0x400000400007FFuLL >> v4) & 1) == 0;
    if ( (~(0x400000400007FFuLL >> v4) & 1) == 0 )
      return v5;
  }
  if ( *(_BYTE *)(*((_QWORD *)a2 + 1) + 8LL) == 7 )
    return 0;
  if ( (unsigned __int8)sub_B46420((__int64)a2) )
    return 0;
  v5 = sub_B46970(a2);
  if ( (_BYTE)v5 )
    return 0;
  if ( (_BYTE)v3 == 85 )
  {
    v60 = *((_QWORD *)a2 - 4);
    if ( v60
      && !*(_BYTE *)v60
      && *(_QWORD *)(v60 + 24) == *((_QWORD *)a2 + 10)
      && (*(_BYTE *)(v60 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v60 + 36) - 68) <= 3 )
    {
      return v5;
    }
    goto LABEL_87;
  }
  if ( (unsigned __int8)(v3 - 82) <= 1u || (_BYTE)v3 == 63 )
    return v5;
  v7 = (unsigned int)(v3 - 34);
  if ( (unsigned __int8)v7 <= 0x33u )
  {
    v77 = 0x8000000000041LL;
    if ( _bittest64(&v77, v7) )
    {
      v60 = *((_QWORD *)a2 - 4);
LABEL_87:
      if ( *(_BYTE *)v60 == 25 )
        return v5;
    }
  }
  v79 = a1 + 136;
  v81 = sub_278A710(a1 + 136, (__int64)a2, 1);
  v89 = *((_QWORD *)a2 + 5);
  if ( *(_BYTE *)(a1 + 760) )
    sub_2798350(a1, *(_QWORD *)(*((_QWORD *)a2 + 5) + 72LL));
  v98 = v100;
  v99 = 0x800000000LL;
  v8 = *(_QWORD *)(v89 + 16);
  if ( !v8 )
    goto LABEL_77;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v8 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
      break;
    v8 = *(_QWORD *)(v8 + 8);
    if ( !v8 )
      goto LABEL_77;
  }
  v10 = v8;
  v78 = 0;
  v80 = 0;
  v85 = 0;
LABEL_17:
  v11 = *(_QWORD *)(v9 + 40);
  v12 = *(_QWORD *)(a1 + 24);
  if ( v11 )
  {
    v13 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
    v14 = *(_DWORD *)(v11 + 44) + 1;
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  if ( v14 >= *(_DWORD *)(v12 + 32) || !*(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v13) )
    goto LABEL_79;
  v93 = v11;
  v92[0] = 0;
  v92[1] = 0;
  if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
    sub_BD73F0((__int64)v92);
  v15 = (unsigned int *)sub_2796930(a1 + 728, (__int64)v92);
  v16 = a1 + 728;
  v17 = *v15;
  v94 = 0;
  v96 = (const char *)v89;
  v95 = 0;
  if ( v89 != -8192 && v89 != -4096 )
  {
    sub_BD73F0((__int64)&v94);
    v16 = a1 + 728;
  }
  v18 = *(_DWORD *)sub_2796930(v16, (__int64)&v94);
  if ( v96 + 4096 != 0 && v96 != 0 && v96 != (const char *)-8192LL )
  {
    v82 = v18;
    sub_BD60C0(&v94);
    v18 = v82;
  }
  if ( v93 != 0 && v93 != -4096 && v93 != -8192 )
  {
    v83 = v18;
    sub_BD60C0(v92);
    v18 = v83;
  }
  if ( v18 <= v17 )
    goto LABEL_79;
  v19 = sub_2797350(v79, v11, v89, v81);
  v20 = sub_278BCD0(a1, v11, v19);
  v23 = v20;
  if ( v20 )
  {
    if ( v20 != a2 )
    {
      v24 = (unsigned int)v99;
      v25 = (unsigned int)v99 + 1LL;
      if ( v25 > HIDWORD(v99) )
      {
        sub_C8D5F0((__int64)&v98, v100, v25, 0x10u, v21, v22);
        v24 = (unsigned int)v99;
      }
      v26 = (unsigned __int8 **)&v98[16 * v24];
      ++v85;
      *v26 = v23;
      v26[1] = (unsigned __int8 *)v11;
      LODWORD(v99) = v99 + 1;
      goto LABEL_39;
    }
LABEL_79:
    v5 = 0;
    goto LABEL_77;
  }
  v57 = (unsigned int)v99;
  v58 = (unsigned int)v99 + 1LL;
  if ( v58 > HIDWORD(v99) )
  {
    sub_C8D5F0((__int64)&v98, v100, v58, 0x10u, v21, v22);
    v57 = (unsigned int)v99;
  }
  v59 = &v98[16 * v57];
  ++v80;
  *v59 = 0;
  v59[1] = v11;
  v78 = v11;
  LODWORD(v99) = v99 + 1;
LABEL_39:
  while ( 1 )
  {
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      break;
    v9 = *(_QWORD *)(v10 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
      goto LABEL_17;
  }
  v5 = 0;
  if ( v85 == 0 || v80 > 1 )
    goto LABEL_77;
  if ( !v80 )
  {
    v87 = 0;
    goto LABEL_51;
  }
  if ( !sub_991A70(a2, 0, 0, 0, 0, 1u, 0) && (unsigned __int8)sub_30ED170(*(_QWORD *)(a1 + 104), a2) )
    goto LABEL_77;
  v28 = *(_QWORD *)(v78 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v28 == v78 + 48 )
    goto LABEL_120;
  if ( !v28 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 > 0xA )
LABEL_120:
    BUG();
  if ( *(_BYTE *)(v28 - 24) == 33 )
    goto LABEL_77;
  v29 = (unsigned int)sub_D0E820(v78, v89);
  v30 = sub_986580(v78);
  if ( (unsigned __int8)sub_D0E970(v30, v29, 0) )
  {
    v72 = sub_986580(v78);
    v74 = *(unsigned int *)(a1 + 776);
    v75 = v72;
    if ( v74 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 780) )
    {
      sub_C8D5F0(a1 + 768, (const void *)(a1 + 784), v74 + 1, 0x10u, v74 + 1, v73);
      v74 = *(unsigned int *)(a1 + 776);
    }
    v76 = (unsigned __int64 *)(*(_QWORD *)(a1 + 768) + 16 * v74);
    v5 = 0;
    *v76 = v75;
    v76[1] = v29;
    ++*(_DWORD *)(a1 + 776);
    goto LABEL_77;
  }
  v31 = sub_B47F80(a2);
  v87 = v31;
  v5 = sub_2797D50(a1, v31, v78, v89);
  if ( (_BYTE)v5 )
  {
    v32 = *(_QWORD *)(a1 + 16);
    if ( v32 )
      sub_102BD20(v32, v31);
LABEL_51:
    v33 = sub_BD5D20((__int64)a2);
    v34 = *((_QWORD *)a2 + 1);
    v94 = (char *)v33;
    v35 = v99;
    v97 = 773;
    v95 = v36;
    v96 = ".pre-phi";
    v37 = sub_BD2DA0(80);
    v38 = v37;
    if ( v37 )
    {
      v39 = (_QWORD *)v37;
      sub_B44260(v37, v34, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v38 + 72) = v35;
      sub_BD6B50((unsigned __int8 *)v38, (const char **)&v94);
      sub_BD2A10(v38, *(_DWORD *)(v38 + 72), 1);
    }
    else
    {
      v39 = 0;
    }
    sub_B44220(v39, *(_QWORD *)(v89 + 56), 1);
    v91 = 16LL * (unsigned int)v99;
    v86 = v87 + 16;
    if ( (_DWORD)v99 )
    {
      v84 = v2;
      v40 = v38;
      v41 = 0;
      v42 = v87;
      do
      {
        v50 = *(_QWORD *)&v98[v41];
        if ( v50 )
        {
          sub_F57050(a2, *(unsigned __int8 **)&v98[v41]);
          v43 = *(_QWORD *)&v98[v41 + 8];
          v44 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
          if ( v44 == *(_DWORD *)(v40 + 72) )
          {
            v88 = *(_QWORD *)&v98[v41 + 8];
            sub_B48D90(v40);
            v43 = v88;
            v44 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
          }
          v45 = (v44 + 1) & 0x7FFFFFF;
          v46 = v45 | *(_DWORD *)(v40 + 4) & 0xF8000000;
          v47 = *(_QWORD *)(v40 - 8) + 32LL * (unsigned int)(v45 - 1);
          *(_DWORD *)(v40 + 4) = v46;
          if ( *(_QWORD *)v47 )
          {
            v48 = *(_QWORD *)(v47 + 8);
            **(_QWORD **)(v47 + 16) = v48;
            if ( v48 )
              *(_QWORD *)(v48 + 16) = *(_QWORD *)(v47 + 16);
          }
          *(_QWORD *)v47 = v50;
          v49 = *(_QWORD *)(v50 + 16);
          *(_QWORD *)(v47 + 8) = v49;
          if ( v49 )
            *(_QWORD *)(v49 + 16) = v47 + 8;
          *(_QWORD *)(v47 + 16) = v50 + 16;
          *(_QWORD *)(v50 + 16) = v47;
          *(_QWORD *)(*(_QWORD *)(v40 - 8)
                    + 32LL * *(unsigned int *)(v40 + 72)
                    + 8LL * ((*(_DWORD *)(v40 + 4) & 0x7FFFFFFu) - 1)) = v43;
        }
        else
        {
          v51 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
          if ( v51 == *(_DWORD *)(v40 + 72) )
          {
            sub_B48D90(v40);
            v51 = *(_DWORD *)(v40 + 4) & 0x7FFFFFF;
          }
          v52 = (v51 + 1) & 0x7FFFFFF;
          v53 = v52 | *(_DWORD *)(v40 + 4) & 0xF8000000;
          v54 = *(_QWORD *)(v40 - 8) + 32LL * (unsigned int)(v52 - 1);
          *(_DWORD *)(v40 + 4) = v53;
          if ( *(_QWORD *)v54 )
          {
            v55 = *(_QWORD *)(v54 + 8);
            **(_QWORD **)(v54 + 16) = v55;
            if ( v55 )
              *(_QWORD *)(v55 + 16) = *(_QWORD *)(v54 + 16);
          }
          *(_QWORD *)v54 = v42;
          if ( v42 )
          {
            v56 = *(_QWORD *)(v42 + 16);
            *(_QWORD *)(v54 + 8) = v56;
            if ( v56 )
              *(_QWORD *)(v56 + 16) = v54 + 8;
            *(_QWORD *)(v54 + 16) = v86;
            *(_QWORD *)(v42 + 16) = v54;
          }
          *(_QWORD *)(*(_QWORD *)(v40 - 8)
                    + 32LL * *(unsigned int *)(v40 + 72)
                    + 8LL * ((*(_DWORD *)(v40 + 4) & 0x7FFFFFFu) - 1)) = v78;
        }
        v41 += 16;
      }
      while ( v91 != v41 );
      v38 = v40;
      v2 = v84;
    }
    sub_2790CB0(v79, (_BYTE *)v38, v81);
    sub_278BB50(v79, v81, v89);
    v61 = (const char **)(v38 + 48);
    sub_27915B0((__int64)(v2 + 44), v81, v38, v89);
    v62 = (const char *)*((_QWORD *)a2 + 6);
    v94 = (char *)v62;
    if ( v62 )
    {
      sub_B96E90((__int64)&v94, (__int64)v62, 1);
      if ( v61 == (const char **)&v94 )
      {
        if ( v94 )
          sub_B91220((__int64)&v94, (__int64)v94);
        goto LABEL_94;
      }
      v70 = *(_QWORD *)(v38 + 48);
      if ( !v70 )
      {
LABEL_104:
        v71 = (unsigned __int8 *)v94;
        *(_QWORD *)(v38 + 48) = v94;
        if ( v71 )
          sub_B976B0((__int64)&v94, v71, v38 + 48);
        goto LABEL_94;
      }
    }
    else if ( v61 == (const char **)&v94 || (v70 = *(_QWORD *)(v38 + 48)) == 0 )
    {
LABEL_94:
      sub_BD84D0((__int64)a2, v38);
      v63 = v2[2];
      if ( v63 )
      {
        v64 = *(_QWORD *)(v38 + 8);
        v65 = *(unsigned __int8 *)(v64 + 8);
        if ( (unsigned int)(v65 - 17) <= 1 )
          LOBYTE(v65) = *(_BYTE *)(**(_QWORD **)(v64 + 16) + 8LL);
        if ( (_BYTE)v65 == 14 )
          sub_102B9D0(v63, v38);
      }
      v5 = 1;
      sub_278A7A0(v79, a2);
      sub_27918D0((__int64)(v2 + 44), v81, (__int64)a2, v89);
      sub_278C2C0(v2, a2, v66, v67, v68, v69);
      goto LABEL_77;
    }
    sub_B91220(v38 + 48, v70);
    goto LABEL_104;
  }
  sub_BD72D0(v31, v31);
LABEL_77:
  if ( v98 != v100 )
    _libc_free((unsigned __int64)v98);
  return v5;
}
