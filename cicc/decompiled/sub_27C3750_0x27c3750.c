// Function: sub_27C3750
// Address: 0x27c3750
//
void __fastcall sub_27C3750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r12
  __int64 v23; // rdx
  int v24; // eax
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rcx
  unsigned __int64 *v28; // r15
  unsigned int i; // eax
  __int64 *v30; // rdx
  __int64 v31; // r12
  __int64 *v32; // rax
  __int64 v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // r8
  __int64 v43; // rsi
  int v44; // ecx
  __int64 v45; // rdi
  int v46; // ecx
  unsigned int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // r11
  _QWORD *v50; // rdx
  __int64 *v51; // rax
  __int64 v52; // r11
  _QWORD *v53; // rax
  __int64 v54; // r8
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 v57; // r15
  __int64 v58; // rbx
  __int64 v59; // r12
  __int64 v60; // rdx
  int v61; // eax
  _QWORD *v62; // rdi
  char v63; // dl
  unsigned __int64 *v64; // r12
  unsigned __int64 *v65; // rdi
  int v66; // eax
  int v67; // eax
  int v68; // eax
  unsigned __int64 *v69; // r13
  unsigned __int64 *v70; // rdi
  __int32 v71; // r12d
  __int64 v73; // [rsp+8h] [rbp-198h]
  __int64 v75; // [rsp+28h] [rbp-178h]
  unsigned __int64 *v76; // [rsp+28h] [rbp-178h]
  __int64 v77; // [rsp+30h] [rbp-170h]
  __int64 v78; // [rsp+38h] [rbp-168h]
  int v79; // [rsp+38h] [rbp-168h]
  __int64 *v80; // [rsp+40h] [rbp-160h] BYREF
  __int64 v81; // [rsp+48h] [rbp-158h]
  _BYTE v82[48]; // [rsp+50h] [rbp-150h] BYREF
  __m128i v83; // [rsp+80h] [rbp-120h] BYREF
  __int64 v84; // [rsp+90h] [rbp-110h]
  __int64 v85; // [rsp+98h] [rbp-108h]
  __int64 v86; // [rsp+A0h] [rbp-100h]
  __int64 v87; // [rsp+A8h] [rbp-F8h]
  __int64 v88; // [rsp+B0h] [rbp-F0h]
  __int64 v89; // [rsp+B8h] [rbp-E8h]
  __int16 v90; // [rsp+C0h] [rbp-E0h]
  unsigned __int64 v91; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 *v92; // [rsp+D8h] [rbp-C8h]
  __int64 v93; // [rsp+E0h] [rbp-C0h]
  int v94; // [rsp+E8h] [rbp-B8h]
  unsigned __int8 v95; // [rsp+ECh] [rbp-B4h]
  char v96; // [rsp+F0h] [rbp-B0h] BYREF

  v6 = sub_D4B130(a2);
  v7 = **(_QWORD **)(a2 + 32);
  v80 = (__int64 *)v82;
  v81 = 0x600000000LL;
  v10 = sub_AA5930(v7);
  v75 = v11;
  v73 = a3 + 16;
  if ( v10 != v11 )
  {
    v12 = v10;
    v13 = v6;
    v14 = v12;
    do
    {
      v15 = *(_QWORD *)(v14 - 8);
      v16 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) != 0 )
      {
        v17 = 0;
        do
        {
          if ( v13 == *(_QWORD *)(v15 + 32LL * *(unsigned int *)(v14 + 72) + 8 * v17) )
          {
            v16 = 32 * v17;
            goto LABEL_8;
          }
          ++v17;
        }
        while ( (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) != (_DWORD)v17 );
        v16 = 0x1FFFFFFFE0LL;
      }
LABEL_8:
      v18 = *(_QWORD *)(v15 + v16);
      v19 = (unsigned int)v81;
      if ( *(_QWORD *)(v14 + 16) )
      {
        v78 = a3;
        v20 = *(_QWORD *)(v14 + 16);
        v77 = a4;
        v21 = *(_QWORD *)(v15 + v16);
        do
        {
          v22 = *(_QWORD *)(v20 + 24);
          if ( v19 + 1 > (unsigned __int64)HIDWORD(v81) )
          {
            sub_C8D5F0((__int64)&v80, v82, v19 + 1, 8u, v19 + 1, v9);
            v19 = (unsigned int)v81;
          }
          v80[v19] = v22;
          v19 = (unsigned int)(v81 + 1);
          LODWORD(v81) = v81 + 1;
          v20 = *(_QWORD *)(v20 + 8);
        }
        while ( v20 );
        v18 = v21;
        a3 = v78;
        a4 = v77;
      }
      sub_DAC8D0(a4, (_BYTE *)v14);
      sub_BD84D0(v14, v18);
      v23 = *(unsigned int *)(a3 + 8);
      v24 = v23;
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v23 )
      {
        v64 = (unsigned __int64 *)sub_C8D7D0(a3, v73, 0, 0x18u, &v91, v9);
        v65 = &v64[3 * *(unsigned int *)(a3 + 8)];
        if ( v65 )
        {
          *v65 = 6;
          v65[1] = 0;
          v65[2] = v14;
          if ( v14 != -4096 && v14 != -8192 )
            sub_BD73F0((__int64)v65);
        }
        sub_F17F80(a3, v64);
        v66 = v91;
        if ( v73 != *(_QWORD *)a3 )
        {
          v79 = v91;
          _libc_free(*(_QWORD *)a3);
          v66 = v79;
        }
        ++*(_DWORD *)(a3 + 8);
        *(_QWORD *)a3 = v64;
        *(_DWORD *)(a3 + 12) = v66;
      }
      else
      {
        v25 = (_QWORD *)(*(_QWORD *)a3 + 24 * v23);
        if ( v25 )
        {
          *v25 = 6;
          v25[1] = 0;
          v25[2] = v14;
          if ( v14 != -4096 && v14 != -8192 )
            sub_BD73F0((__int64)v25);
          v24 = *(_DWORD *)(a3 + 8);
        }
        *(_DWORD *)(a3 + 8) = v24 + 1;
      }
      v26 = *(_QWORD *)(v14 + 32);
      if ( !v26 )
        BUG();
      v14 = 0;
      if ( *(_BYTE *)(v26 - 24) == 84 )
        v14 = v26 - 24;
    }
    while ( v75 != v14 );
  }
  v27 = 1;
  v91 = 0;
  v28 = &v91;
  v92 = (__int64 *)&v96;
  v93 = 16;
  v94 = 0;
  v95 = 1;
LABEL_26:
  for ( i = v81; (_DWORD)v81; i = v81 )
  {
    v30 = v80;
    v31 = v80[i - 1];
    LODWORD(v81) = i - 1;
    if ( !(_BYTE)v27 )
      goto LABEL_68;
    v32 = v92;
    v30 = &v92[HIDWORD(v93)];
    if ( v92 != v30 )
    {
      while ( v31 != *v32 )
      {
        if ( v30 == ++v32 )
          goto LABEL_31;
      }
      goto LABEL_26;
    }
LABEL_31:
    if ( HIDWORD(v93) < (unsigned int)v93 )
    {
      ++HIDWORD(v93);
      *v30 = v31;
      v27 = v95;
      ++v91;
    }
    else
    {
LABEL_68:
      sub_C8CC70((__int64)v28, v31, (__int64)v30, v27, v8, v9);
      v27 = v95;
      if ( !v63 )
        goto LABEL_26;
    }
    v33 = *(_QWORD *)(v31 + 40);
    if ( *(_BYTE *)(a2 + 84) )
    {
      v34 = *(_QWORD **)(a2 + 64);
      v35 = &v34[*(unsigned int *)(a2 + 76)];
      if ( v34 == v35 )
        goto LABEL_26;
      while ( v33 != *v34 )
      {
        if ( v35 == ++v34 )
          goto LABEL_26;
      }
    }
    else if ( !sub_C8CA60(a2 + 56, v33) )
    {
      v27 = v95;
      continue;
    }
    v83 = (__m128i)(unsigned __int64)sub_B43CC0(v31);
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v87 = 0;
    v88 = 0;
    v89 = 0;
    v90 = 257;
    v40 = sub_1020E10(v31, &v83, v36, v37, v38, v39);
    v41 = v40;
    if ( v40 )
    {
      if ( *(_BYTE *)v40 > 0x1Cu )
      {
        v42 = *(_QWORD *)(v40 + 40);
        v43 = *(_QWORD *)(v31 + 40);
        if ( v42 != v43 )
        {
          v44 = *(_DWORD *)(a1 + 24);
          v45 = *(_QWORD *)(a1 + 8);
          if ( v44 )
          {
            v46 = v44 - 1;
            v47 = v46 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
            v48 = (__int64 *)(v45 + 16LL * v47);
            v49 = *v48;
            if ( *v48 == v42 )
            {
LABEL_43:
              v50 = (_QWORD *)v48[1];
              if ( v50 )
              {
                v8 = v46 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v51 = (__int64 *)(v45 + 16 * v8);
                v52 = *v51;
                if ( v43 != *v51 )
                {
                  v68 = 1;
                  while ( v52 != -4096 )
                  {
                    v9 = (unsigned int)(v68 + 1);
                    v8 = v46 & (unsigned int)(v68 + v8);
                    v51 = (__int64 *)(v45 + 16LL * (unsigned int)v8);
                    v52 = *v51;
                    if ( v43 == *v51 )
                      goto LABEL_45;
                    v68 = v9;
                  }
                  goto LABEL_61;
                }
LABEL_45:
                v53 = (_QWORD *)v51[1];
                if ( v50 != v53 )
                {
                  while ( v53 )
                  {
                    v53 = (_QWORD *)*v53;
                    if ( v50 == v53 )
                      goto LABEL_48;
                  }
                  goto LABEL_61;
                }
              }
            }
            else
            {
              v67 = 1;
              while ( v49 != -4096 )
              {
                v9 = (unsigned int)(v67 + 1);
                v47 = v46 & (v67 + v47);
                v48 = (__int64 *)(v45 + 16LL * v47);
                v49 = *v48;
                if ( v42 == *v48 )
                  goto LABEL_43;
                v67 = v9;
              }
            }
          }
        }
      }
LABEL_48:
      v54 = *(_QWORD *)(v31 + 16);
      v55 = (unsigned int)v81;
      if ( v54 )
      {
        v56 = v31;
        v76 = v28;
        v57 = a3;
        v58 = *(_QWORD *)(v31 + 16);
        do
        {
          v59 = *(_QWORD *)(v58 + 24);
          if ( v55 + 1 > (unsigned __int64)HIDWORD(v81) )
          {
            sub_C8D5F0((__int64)&v80, v82, v55 + 1, 8u, v54, v9);
            v55 = (unsigned int)v81;
          }
          v80[v55] = v59;
          v55 = (unsigned int)(v81 + 1);
          LODWORD(v81) = v81 + 1;
          v58 = *(_QWORD *)(v58 + 8);
        }
        while ( v58 );
        v31 = v56;
        a3 = v57;
        v28 = v76;
      }
      sub_BD84D0(v31, v41);
      v60 = *(unsigned int *)(a3 + 8);
      v61 = v60;
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v60 )
      {
        v69 = (unsigned __int64 *)sub_C8D7D0(a3, a3 + 16, 0, 0x18u, (unsigned __int64 *)&v83, v9);
        v70 = &v69[3 * *(unsigned int *)(a3 + 8)];
        if ( v70 )
        {
          *v70 = 6;
          v70[1] = 0;
          v70[2] = v31;
          if ( v31 != -4096 && v31 != -8192 )
            sub_BD73F0((__int64)v70);
        }
        sub_F17F80(a3, v69);
        v71 = v83.m128i_i32[0];
        if ( a3 + 16 != *(_QWORD *)a3 )
          _libc_free(*(_QWORD *)a3);
        ++*(_DWORD *)(a3 + 8);
        *(_QWORD *)a3 = v69;
        *(_DWORD *)(a3 + 12) = v71;
      }
      else
      {
        v62 = (_QWORD *)(*(_QWORD *)a3 + 24 * v60);
        if ( v62 )
        {
          *v62 = 6;
          v62[1] = 0;
          v62[2] = v31;
          if ( v31 != -4096 && v31 != -8192 )
            sub_BD73F0((__int64)v62);
          v61 = *(_DWORD *)(a3 + 8);
        }
        *(_DWORD *)(a3 + 8) = v61 + 1;
      }
    }
LABEL_61:
    v27 = v95;
  }
  if ( !(_BYTE)v27 )
    _libc_free((unsigned __int64)v92);
  if ( v80 != (__int64 *)v82 )
    _libc_free((unsigned __int64)v80);
}
