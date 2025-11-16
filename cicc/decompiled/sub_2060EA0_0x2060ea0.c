// Function: sub_2060EA0
// Address: 0x2060ea0
//
void __fastcall sub_2060EA0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v7; // r12
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // r13
  __int64 v22; // r11
  __int64 v23; // r9
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 *v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  _QWORD *v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rcx
  __int64 v37; // r8
  int v38; // r9d
  __int16 *v39; // rdx
  __int64 *v40; // r8
  __int16 *v41; // r9
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 *v44; // r12
  int v45; // edx
  int v46; // r14d
  __int64 v47; // r13
  int v48; // eax
  __int64 v49; // rdi
  unsigned int v50; // eax
  __int64 v51; // rsi
  int v52; // r10d
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r14
  _QWORD *v58; // r12
  __int64 v59; // rdx
  __int64 v60; // r13
  __int64 v61; // rcx
  __int64 v62; // r8
  int v63; // r9d
  unsigned __int64 v64; // rdx
  __int64 *v65; // r8
  unsigned __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // rsi
  int v69; // edx
  __int64 *v70; // rbx
  int v71; // r12d
  int v72; // r11d
  _QWORD *v73; // rdx
  int v74; // eax
  int v75; // ecx
  int v76; // eax
  int v77; // esi
  __int64 v78; // rdi
  unsigned int v79; // eax
  __int64 v80; // r8
  int v81; // r10d
  _QWORD *v82; // r9
  int v83; // eax
  int v84; // eax
  __int64 v85; // rdi
  _QWORD *v86; // r8
  unsigned int v87; // r15d
  int v88; // r9d
  __int64 v89; // rsi
  int v90; // eax
  int v91; // eax
  int v92; // eax
  __int64 v93; // rsi
  __int64 v94; // rdi
  unsigned int v95; // r14d
  int v96; // edx
  __int64 *v97; // r11
  __int128 v98; // [rsp-20h] [rbp-C0h]
  __int128 v99; // [rsp-10h] [rbp-B0h]
  __int64 *v100; // [rsp+0h] [rbp-A0h]
  __int16 *v101; // [rsp+8h] [rbp-98h]
  _QWORD *v102; // [rsp+10h] [rbp-90h]
  __int64 *v103; // [rsp+10h] [rbp-90h]
  __int64 v104; // [rsp+18h] [rbp-88h]
  unsigned __int64 v105; // [rsp+18h] [rbp-88h]
  __int64 v106; // [rsp+60h] [rbp-40h] BYREF
  int v107; // [rsp+68h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(_QWORD *)(a2 - 24);
  v9 = *(_DWORD *)(v7 + 72);
  if ( !v9 )
  {
    ++*(_QWORD *)(v7 + 48);
LABEL_54:
    sub_1D52F30(v7 + 48, 2 * v9);
    v76 = *(_DWORD *)(v7 + 72);
    if ( !v76 )
      goto LABEL_114;
    v77 = v76 - 1;
    v78 = *(_QWORD *)(v7 + 56);
    v79 = (v76 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v75 = *(_DWORD *)(v7 + 64) + 1;
    v73 = (_QWORD *)(v78 + 16LL * v79);
    v80 = *v73;
    if ( v8 != *v73 )
    {
      v81 = 1;
      v82 = 0;
      while ( v80 != -8 )
      {
        if ( !v82 && v80 == -16 )
          v82 = v73;
        v79 = v77 & (v81 + v79);
        v73 = (_QWORD *)(v78 + 16LL * v79);
        v80 = *v73;
        if ( v8 == *v73 )
          goto LABEL_50;
        ++v81;
      }
      if ( v82 )
        v73 = v82;
    }
    goto LABEL_50;
  }
  v10 = *(_QWORD *)(v7 + 56);
  v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v12 = (_QWORD *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( v8 == *v12 )
  {
    v14 = v12[1];
    goto LABEL_4;
  }
  v72 = 1;
  v73 = 0;
  while ( v13 != -8 )
  {
    if ( v13 != -16 || v73 )
      v12 = v73;
    v96 = v72 + 1;
    v11 = (v9 - 1) & (v72 + v11);
    v97 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v97;
    if ( v8 == *v97 )
    {
      v14 = v97[1];
      goto LABEL_4;
    }
    v72 = v96;
    v73 = v12;
    v12 = (_QWORD *)(v10 + 16LL * v11);
  }
  if ( !v73 )
    v73 = v12;
  v74 = *(_DWORD *)(v7 + 64);
  ++*(_QWORD *)(v7 + 48);
  v75 = v74 + 1;
  if ( 4 * (v74 + 1) >= 3 * v9 )
    goto LABEL_54;
  if ( v9 - *(_DWORD *)(v7 + 68) - v75 <= v9 >> 3 )
  {
    sub_1D52F30(v7 + 48, v9);
    v83 = *(_DWORD *)(v7 + 72);
    if ( !v83 )
      goto LABEL_114;
    v84 = v83 - 1;
    v85 = *(_QWORD *)(v7 + 56);
    v86 = 0;
    v87 = v84 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v88 = 1;
    v75 = *(_DWORD *)(v7 + 64) + 1;
    v73 = (_QWORD *)(v85 + 16LL * v87);
    v89 = *v73;
    if ( v8 != *v73 )
    {
      while ( v89 != -8 )
      {
        if ( v89 == -16 && !v86 )
          v86 = v73;
        v87 = v84 & (v88 + v87);
        v73 = (_QWORD *)(v85 + 16LL * v87);
        v89 = *v73;
        if ( v8 == *v73 )
          goto LABEL_50;
        ++v88;
      }
      if ( v86 )
        v73 = v86;
    }
  }
LABEL_50:
  *(_DWORD *)(v7 + 64) = v75;
  if ( *v73 != -8 )
    --*(_DWORD *)(v7 + 68);
  *v73 = v8;
  v14 = 0;
  v73[1] = 0;
  v7 = *(_QWORD *)(a1 + 712);
LABEL_4:
  sub_1DD8FE0(*(_QWORD *)(v7 + 784), v14, -1);
  v15 = sub_15E38F0(**(_QWORD **)(a1 + 712));
  if ( (unsigned int)sub_14DD7D0(v15) - 7 > 1 )
  {
    v18 = *(_QWORD *)(*(_QWORD *)(a2 - 48) - 24LL);
    if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
    {
      v7 = *(_QWORD *)(a1 + 712);
      v19 = **(_QWORD **)(v18 - 8);
      if ( *(_BYTE *)(v19 + 16) != 16 )
      {
LABEL_7:
        v20 = *(_DWORD *)(v7 + 72);
        v21 = *(_QWORD *)(v19 + 40);
        v22 = v7 + 48;
        if ( v20 )
          goto LABEL_8;
        goto LABEL_22;
      }
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 712);
      v16 = 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
      v19 = *(_QWORD *)(v18 - v16);
      if ( *(_BYTE *)(v19 + 16) != 16 )
        goto LABEL_7;
    }
    v20 = *(_DWORD *)(v7 + 72);
    v22 = v7 + 48;
    v21 = *(_QWORD *)(*(_QWORD *)v7 + 80LL);
    if ( v21 )
      v21 -= 24;
    if ( v20 )
    {
LABEL_8:
      v23 = *(_QWORD *)(v7 + 56);
      v24 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v25 = (__int64 *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( *v25 == v21 )
      {
        v27 = v25[1];
LABEL_10:
        v28 = *(__int64 **)(a1 + 552);
        v102 = sub_1D2A490(v28, v27, v16, v26, v17, v23);
        v104 = v29;
        v33 = sub_1D2A490(*(_QWORD **)(a1 + 552), v14, v29, v30, v31, v32);
        v35 = v34;
        v106 = 0;
        v40 = sub_2051DF0((__int64 *)a1, *(double *)a3.m128_u64, a4, a5, v14, v34, v36, v37, v38);
        v41 = v39;
        v42 = *(_QWORD *)a1;
        v107 = *(_DWORD *)(a1 + 536);
        if ( v42 )
        {
          if ( &v106 != (__int64 *)(v42 + 48) )
          {
            v43 = *(_QWORD *)(v42 + 48);
            v106 = v43;
            if ( v43 )
            {
              v100 = v40;
              v101 = v39;
              sub_1623A60((__int64)&v106, v43, 2);
              v40 = v100;
              v41 = v101;
            }
          }
        }
        *((_QWORD *)&v98 + 1) = v35;
        *(_QWORD *)&v98 = v33;
        v44 = sub_1D3A900(
                v28,
                0xC5u,
                (__int64)&v106,
                1u,
                0,
                0,
                a3,
                a4,
                a5,
                (unsigned __int64)v40,
                v41,
                v98,
                (__int64)v102,
                v104);
        v46 = v45;
        if ( v106 )
          sub_161E7C0((__int64)&v106, v106);
        v47 = *(_QWORD *)(a1 + 552);
        if ( v44 )
        {
          nullsub_686();
          *(_QWORD *)(v47 + 176) = v44;
          *(_DWORD *)(v47 + 184) = v46;
          sub_1D23870();
        }
        else
        {
          *(_QWORD *)(v47 + 176) = 0;
          *(_DWORD *)(v47 + 184) = v46;
        }
        return;
      }
      v17 = 1;
      v16 = 0;
      while ( v26 != -8 )
      {
        if ( v26 != -16 || v16 )
          v25 = (__int64 *)v16;
        v16 = (unsigned int)(v17 + 1);
        v24 = (v20 - 1) & (v17 + v24);
        v17 = v23 + 16LL * v24;
        v26 = *(_QWORD *)v17;
        if ( *(_QWORD *)v17 == v21 )
        {
          v27 = *(_QWORD *)(v17 + 8);
          goto LABEL_10;
        }
        v17 = (unsigned int)v16;
        v16 = (__int64)v25;
        v25 = (__int64 *)(v23 + 16LL * v24);
      }
      if ( !v16 )
        v16 = (__int64)v25;
      v90 = *(_DWORD *)(v7 + 64);
      ++*(_QWORD *)(v7 + 48);
      if ( 4 * (v90 + 1) < 3 * v20 )
      {
        v26 = v20 >> 3;
        if ( v20 - *(_DWORD *)(v7 + 68) - (v90 + 1) > (unsigned int)v26 )
        {
LABEL_73:
          ++*(_DWORD *)(v7 + 64);
          if ( *(_QWORD *)v16 != -8 )
            --*(_DWORD *)(v7 + 68);
          *(_QWORD *)v16 = v21;
          v27 = 0;
          *(_QWORD *)(v16 + 8) = 0;
          goto LABEL_10;
        }
        sub_1D52F30(v22, v20);
        v91 = *(_DWORD *)(v7 + 72);
        if ( v91 )
        {
          v92 = v91 - 1;
          v93 = *(_QWORD *)(v7 + 56);
          v23 = 1;
          v94 = 0;
          v95 = v92 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v16 = v93 + 16LL * v95;
          v26 = *(_QWORD *)v16;
          if ( *(_QWORD *)v16 != v21 )
          {
            while ( v26 != -8 )
            {
              if ( v26 == -16 && !v94 )
                v94 = v16;
              v95 = v92 & (v23 + v95);
              v16 = v93 + 16LL * v95;
              v26 = *(_QWORD *)v16;
              if ( *(_QWORD *)v16 == v21 )
                goto LABEL_73;
              v23 = (unsigned int)(v23 + 1);
            }
            if ( v94 )
              v16 = v94;
          }
          goto LABEL_73;
        }
LABEL_114:
        ++*(_DWORD *)(v7 + 64);
        BUG();
      }
LABEL_23:
      sub_1D52F30(v22, 2 * v20);
      v48 = *(_DWORD *)(v7 + 72);
      if ( v48 )
      {
        v26 = (unsigned int)(v48 - 1);
        v49 = *(_QWORD *)(v7 + 56);
        v50 = v26 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v16 = v49 + 16LL * v50;
        v51 = *(_QWORD *)v16;
        if ( *(_QWORD *)v16 != v21 )
        {
          v52 = 1;
          v23 = 0;
          while ( v51 != -8 )
          {
            if ( !v23 && v51 == -16 )
              v23 = v16;
            v50 = v26 & (v52 + v50);
            v16 = v49 + 16LL * v50;
            v51 = *(_QWORD *)v16;
            if ( *(_QWORD *)v16 == v21 )
              goto LABEL_73;
            ++v52;
          }
          if ( v23 )
            v16 = v23;
        }
        goto LABEL_73;
      }
      goto LABEL_114;
    }
LABEL_22:
    ++*(_QWORD *)(v7 + 48);
    goto LABEL_23;
  }
  if ( v14 != sub_2054600(a1, *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL))
    || !(unsigned int)sub_1700720(*(_QWORD *)(a1 + 544)) )
  {
    v57 = *(_QWORD *)(a1 + 552);
    v58 = sub_1D2A490((_QWORD *)v57, v14, v53, v54, v55, v56);
    v60 = v59;
    v106 = 0;
    v65 = sub_2051DF0((__int64 *)a1, *(double *)a3.m128_u64, a4, a5, v14, v59, v61, v62, v63);
    v66 = v64;
    v67 = *(_QWORD *)a1;
    v107 = *(_DWORD *)(a1 + 536);
    if ( v67 )
    {
      if ( &v106 != (__int64 *)(v67 + 48) )
      {
        v68 = *(_QWORD *)(v67 + 48);
        v106 = v68;
        if ( v68 )
        {
          v103 = v65;
          v105 = v64;
          sub_1623A60((__int64)&v106, v68, 2);
          v65 = v103;
          v66 = v105;
        }
      }
    }
    *((_QWORD *)&v99 + 1) = v60;
    *(_QWORD *)&v99 = v58;
    v70 = sub_1D332F0(
            (__int64 *)v57,
            188,
            (__int64)&v106,
            1,
            0,
            0,
            *(double *)a3.m128_u64,
            a4,
            a5,
            (__int64)v65,
            v66,
            v99);
    v71 = v69;
    if ( v70 )
    {
      nullsub_686();
      *(_QWORD *)(v57 + 176) = v70;
      *(_DWORD *)(v57 + 184) = v71;
      sub_1D23870();
    }
    else
    {
      *(_QWORD *)(v57 + 176) = 0;
      *(_DWORD *)(v57 + 184) = v69;
    }
    if ( v106 )
      sub_161E7C0((__int64)&v106, v106);
  }
}
