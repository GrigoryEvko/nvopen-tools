// Function: sub_27975D0
// Address: 0x27975d0
//
__int64 __fastcall sub_27975D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r10
  unsigned int v7; // r13d
  unsigned int v10; // esi
  __int64 v11; // r9
  int v12; // r15d
  unsigned int v13; // edi
  unsigned int *v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // rcx
  int v17; // edx
  unsigned int v18; // r15d
  __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // rsi
  unsigned int v23; // eax
  unsigned int v24; // r10d
  unsigned int *v26; // rdx
  int v27; // eax
  int v28; // ecx
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  int v37; // edx
  unsigned int v38; // ecx
  unsigned int v39; // r15d
  char v40; // al
  __int64 v41; // r8
  unsigned int *v42; // rax
  char v43; // al
  int v44; // eax
  int v45; // esi
  __int64 v46; // r9
  unsigned int v47; // edi
  unsigned int v48; // eax
  int v49; // r11d
  unsigned int *v50; // r10
  int v51; // eax
  int v52; // esi
  __int64 v53; // rdi
  unsigned int *v54; // r9
  unsigned int v55; // r15d
  int v56; // r10d
  unsigned int v57; // eax
  unsigned int v58; // esi
  int v59; // eax
  __int64 v60; // r10
  int v61; // eax
  char v62; // al
  __int64 v63; // rcx
  __int64 v64; // r9
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r13
  unsigned int v69; // r15d
  int v70; // eax
  int v71; // eax
  unsigned int v72; // edx
  __int64 v73; // [rsp+8h] [rbp-D8h]
  __int64 v74; // [rsp+8h] [rbp-D8h]
  __int64 v75; // [rsp+8h] [rbp-D8h]
  unsigned int v76; // [rsp+8h] [rbp-D8h]
  unsigned int v77; // [rsp+18h] [rbp-C8h]
  int v78; // [rsp+18h] [rbp-C8h]
  __int64 v79; // [rsp+18h] [rbp-C8h]
  unsigned int v80; // [rsp+18h] [rbp-C8h]
  unsigned int v81; // [rsp+18h] [rbp-C8h]
  __int64 v82; // [rsp+18h] [rbp-C8h]
  __int64 v83; // [rsp+18h] [rbp-C8h]
  __int64 v84; // [rsp+18h] [rbp-C8h]
  __int64 v85; // [rsp+18h] [rbp-C8h]
  __int64 v86; // [rsp+18h] [rbp-C8h]
  __int64 v87; // [rsp+18h] [rbp-C8h]
  __int64 v88; // [rsp+18h] [rbp-C8h]
  __int64 v89; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+28h] [rbp-B8h] BYREF
  int v91; // [rsp+30h] [rbp-B0h] BYREF
  char v92; // [rsp+34h] [rbp-ACh]
  __int64 v93; // [rsp+38h] [rbp-A8h]
  _BYTE *v94; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v95; // [rsp+48h] [rbp-98h]
  _BYTE v96[16]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v97; // [rsp+60h] [rbp-80h]
  int v98; // [rsp+70h] [rbp-70h] BYREF
  char v99; // [rsp+74h] [rbp-6Ch]
  __int64 v100; // [rsp+78h] [rbp-68h]
  _BYTE *v101; // [rsp+80h] [rbp-60h]
  __int64 v102; // [rsp+88h] [rbp-58h]
  _BYTE v103[16]; // [rsp+90h] [rbp-50h] BYREF
  __int64 v104; // [rsp+A0h] [rbp-40h]

  v5 = a1 + 120;
  v7 = a4;
  v10 = *(_DWORD *)(a1 + 144);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_43;
  }
  v11 = *(_QWORD *)(a1 + 128);
  v12 = 37 * a4;
  v13 = (v10 - 1) & (37 * a4);
  v14 = (unsigned int *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( v7 != *v14 )
  {
    v78 = 1;
    v26 = 0;
    while ( v15 != -1 )
    {
      if ( !v26 && v15 == -2 )
        v26 = v14;
      v13 = (v10 - 1) & (v78 + v13);
      v14 = (unsigned int *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == v7 )
        goto LABEL_3;
      ++v78;
    }
    if ( !v26 )
      v26 = v14;
    v27 = *(_DWORD *)(a1 + 136);
    ++*(_QWORD *)(a1 + 120);
    v28 = v27 + 1;
    if ( 4 * (v27 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 140) - v28 > v10 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 136) = v28;
        if ( *v26 != -1 )
          --*(_DWORD *)(a1 + 140);
        *v26 = v7;
        *((_QWORD *)v26 + 1) = 0;
        goto LABEL_23;
      }
      v83 = a5;
      sub_27908B0(v5, v10);
      v51 = *(_DWORD *)(a1 + 144);
      if ( v51 )
      {
        v52 = v51 - 1;
        v53 = *(_QWORD *)(a1 + 128);
        v54 = 0;
        v55 = (v51 - 1) & v12;
        a5 = v83;
        v56 = 1;
        v28 = *(_DWORD *)(a1 + 136) + 1;
        v26 = (unsigned int *)(v53 + 16LL * v55);
        v57 = *v26;
        if ( *v26 != v7 )
        {
          while ( v57 != -1 )
          {
            if ( !v54 && v57 == -2 )
              v54 = v26;
            v55 = v52 & (v56 + v55);
            v26 = (unsigned int *)(v53 + 16LL * v55);
            v57 = *v26;
            if ( *v26 == v7 )
              goto LABEL_20;
            ++v56;
          }
          if ( v54 )
            v26 = v54;
        }
        goto LABEL_20;
      }
LABEL_93:
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
LABEL_43:
    v82 = a5;
    sub_27908B0(v5, 2 * v10);
    v44 = *(_DWORD *)(a1 + 144);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 128);
      a5 = v82;
      v47 = (v44 - 1) & (37 * v7);
      v28 = *(_DWORD *)(a1 + 136) + 1;
      v26 = (unsigned int *)(v46 + 16LL * v47);
      v48 = *v26;
      if ( *v26 != v7 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -1 )
        {
          if ( !v50 && v48 == -2 )
            v50 = v26;
          v47 = v45 & (v49 + v47);
          v26 = (unsigned int *)(v46 + 16LL * v47);
          v48 = *v26;
          if ( *v26 == v7 )
            goto LABEL_20;
          ++v49;
        }
        if ( v50 )
          v26 = v50;
      }
      goto LABEL_20;
    }
    goto LABEL_93;
  }
LABEL_3:
  v16 = *((_QWORD *)v14 + 1);
  if ( !v16 )
  {
LABEL_23:
    v79 = a5;
    if ( !(unsigned __int8)sub_278B930(a1, v7, a3, a5) )
      return v7;
    v30 = *(_QWORD *)(a1 + 96);
    if ( v7 >= (unsigned __int64)((*(_QWORD *)(a1 + 104) - v30) >> 2) )
      return v7;
    v31 = *(unsigned int *)(v30 + 4LL * v7);
    if ( !(_DWORD)v31 )
      return v7;
    v32 = v79;
    v33 = *(_QWORD *)(a1 + 72) + 56 * v31;
    v91 = *(_DWORD *)v33;
    v92 = *(_BYTE *)(v33 + 4);
    v34 = *(_QWORD *)(v33 + 8);
    v94 = v96;
    v93 = v34;
    v95 = 0x400000000LL;
    v35 = *(unsigned int *)(v33 + 24);
    if ( !(_DWORD)v35 )
    {
      v97 = *(_QWORD *)(v33 + 48);
LABEL_28:
      if ( v92 )
      {
        v36 = (unsigned __int64)v94;
        v37 = *(_DWORD *)v94;
        v38 = *((_DWORD *)v94 + 1);
        if ( *(_DWORD *)v94 > v38 )
        {
          *(_DWORD *)v94 = v38;
          v39 = v91;
          *(_DWORD *)(v36 + 4) = v37;
          if ( (v39 >> 8) - 53 <= 1 )
          {
            v87 = v32;
            v71 = sub_B52F50((unsigned __int8)v39);
            v72 = v39;
            v32 = v87;
            LOBYTE(v72) = 0;
            v91 = v71 | v72;
          }
        }
      }
      v73 = v32;
      v40 = sub_278F8C0(a1 + 32, (__int64)&v91, &v89);
      v41 = v73;
      if ( v40 )
      {
        v42 = (unsigned int *)(v89 + 56);
LABEL_34:
        v24 = *v42;
        if ( !*v42
          || v91 == 56 && v24 != v7 && (v80 = *v42, v43 = sub_278B9D0(a1, v7, v24, a2, a3, v41), v24 = v80, !v43) )
        {
          v24 = v7;
        }
        if ( v94 != v96 )
        {
          v81 = v24;
          _libc_free((unsigned __int64)v94);
          return v81;
        }
        return v24;
      }
      v58 = *(_DWORD *)(a1 + 56);
      v59 = *(_DWORD *)(a1 + 48);
      v60 = v89;
      ++*(_QWORD *)(a1 + 32);
      v61 = v59 + 1;
      v90 = v60;
      if ( 4 * v61 >= 3 * v58 )
      {
        v58 *= 2;
        v88 = v73;
      }
      else
      {
        if ( v58 - *(_DWORD *)(a1 + 52) - v61 > v58 >> 3 )
        {
LABEL_58:
          *(_DWORD *)(a1 + 48) = v61;
          v74 = v41;
          v101 = v103;
          v98 = -1;
          v99 = 0;
          v100 = 0;
          v102 = 0x400000000LL;
          v104 = 0;
          v62 = sub_278A2A0(v60, (__int64)&v98);
          v65 = v74;
          if ( !v62 )
            --*(_DWORD *)(a1 + 52);
          v66 = v90;
          if ( v101 != v103 )
          {
            v84 = v90;
            _libc_free((unsigned __int64)v101);
            v65 = v74;
            v66 = v84;
          }
          v75 = v65;
          v85 = v66;
          *(_DWORD *)v66 = v91;
          *(_BYTE *)(v66 + 4) = v92;
          v67 = v93;
          *(_QWORD *)(v66 + 8) = v93;
          sub_2789770(v66 + 16, (__int64)&v94, v67, v63, v65, v64);
          v41 = v75;
          *(_QWORD *)(v85 + 48) = v97;
          v42 = (unsigned int *)(v85 + 56);
          *(_DWORD *)(v85 + 56) = 0;
          goto LABEL_34;
        }
        v88 = v73;
      }
      sub_27929B0(a1 + 32, v58);
      sub_278F8C0(a1 + 32, (__int64)&v91, &v90);
      v60 = v90;
      v41 = v88;
      v61 = *(_DWORD *)(a1 + 48) + 1;
      goto LABEL_58;
    }
    sub_2789770((__int64)&v94, v33 + 16, v35, v7, v79, v29);
    v32 = v79;
    v97 = *(_QWORD *)(v33 + 48);
    if ( !(_DWORD)v95 )
      goto LABEL_28;
    v76 = v7;
    v68 = 0;
    v69 = 0;
    while ( 1 )
    {
      if ( v69 > 1 )
      {
        if ( v91 == 65 || v91 == 64 || v91 == 63 )
          goto LABEL_69;
      }
      else if ( v69 && v91 == 64 )
      {
        goto LABEL_69;
      }
      v86 = v32;
      v70 = sub_2797350(a1, a2, a3, *(_DWORD *)&v94[4 * v68]);
      v32 = v86;
      *(_DWORD *)&v94[4 * v68] = v70;
LABEL_69:
      v68 = v69 + 1;
      v69 = v68;
      if ( (unsigned int)v68 >= (unsigned int)v95 )
      {
        v7 = v76;
        goto LABEL_28;
      }
    }
  }
  v17 = *(_DWORD *)(v16 + 4);
  v18 = 0;
  if ( (v17 & 0x7FFFFFF) == 0 )
    return v7;
  v77 = v7;
  v19 = a1;
  v20 = a3;
  v21 = *((_QWORD *)v14 + 1);
  while ( 1 )
  {
    while ( 1 )
    {
      if ( v20 == *(_QWORD *)(v21 + 40) )
      {
        v22 = *(_QWORD *)(v21 - 8);
        if ( a2 == *(_QWORD *)(v22 + 32LL * *(unsigned int *)(v21 + 72) + 8LL * v18) )
          break;
      }
      if ( (v17 & 0x7FFFFFF) == ++v18 )
        return v77;
    }
    v23 = sub_278A710(v19, *(_QWORD *)(v22 + 32LL * v18), 0);
    if ( v23 )
      break;
    v17 = *(_DWORD *)(v21 + 4);
    if ( (v17 & 0x7FFFFFF) == ++v18 )
      return v77;
  }
  return v23;
}
