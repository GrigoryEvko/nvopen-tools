// Function: sub_14FC3B0
// Address: 0x14fc3b0
//
__int64 *__fastcall sub_14FC3B0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  int v4; // eax
  char v5; // cl
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  const char **v12; // r11
  int v13; // esi
  _QWORD *v14; // rcx
  unsigned int v15; // edx
  _QWORD *v16; // rax
  __int64 v17; // r8
  char v18; // cl
  unsigned int v19; // r13d
  char v20; // di
  __int64 v21; // r14
  unsigned int v22; // esi
  const char *v23; // rax
  unsigned int v25; // edx
  unsigned int v26; // ecx
  unsigned int v27; // r8d
  int v28; // r10d
  _QWORD *v29; // r9
  _QWORD *v30; // rdi
  int v31; // esi
  unsigned int v32; // ecx
  __int64 v33; // r8
  int v34; // r9d
  _QWORD *v35; // rdx
  _QWORD *v36; // rdi
  int v37; // esi
  unsigned int v38; // ecx
  __int64 v39; // r8
  int v40; // r9d
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r12
  int v45; // r13d
  __int64 *v46; // r11
  __int64 v47; // r12
  int v48; // r14d
  __int64 v49; // rdi
  __int64 *v50; // r10
  int v51; // r8d
  _QWORD *v52; // r9
  unsigned int v53; // esi
  _QWORD *v54; // rdx
  __int64 v55; // r13
  unsigned int v56; // esi
  unsigned int v57; // ecx
  _QWORD *v58; // rdx
  __int64 v59; // r13
  int v60; // edx
  __int64 v61; // r13
  int v62; // edx
  __int64 j; // rax
  __int64 v64; // r12
  unsigned int v65; // ebx
  __int64 i; // r10
  __int64 *v67; // r9
  int v68; // edi
  _QWORD *v69; // r8
  unsigned int v70; // esi
  _QWORD *v71; // rax
  __int64 v72; // r13
  unsigned int v73; // r13d
  unsigned int v74; // ecx
  _QWORD *v75; // rax
  __int64 v76; // rsi
  int v77; // eax
  int v78; // eax
  int v79; // [rsp+14h] [rbp-49Ch]
  int v80; // [rsp+14h] [rbp-49Ch]
  const char **v81; // [rsp+20h] [rbp-490h]
  const char **v82; // [rsp+20h] [rbp-490h]
  __int64 v83; // [rsp+20h] [rbp-490h]
  int v84; // [rsp+20h] [rbp-490h]
  int v85; // [rsp+20h] [rbp-490h]
  __int64 v86; // [rsp+28h] [rbp-488h]
  __int64 v89; // [rsp+58h] [rbp-458h] BYREF
  _QWORD v90[32]; // [rsp+60h] [rbp-450h] BYREF
  const char *v91; // [rsp+160h] [rbp-350h] BYREF
  __int64 v92; // [rsp+168h] [rbp-348h]
  _QWORD *v93; // [rsp+170h] [rbp-340h] BYREF
  unsigned int v94; // [rsp+178h] [rbp-338h]
  const char *v95; // [rsp+270h] [rbp-240h] BYREF
  __int64 v96; // [rsp+278h] [rbp-238h]
  _BYTE v97[560]; // [rsp+280h] [rbp-230h] BYREF

  if ( (unsigned __int8)sub_15127D0(a2 + 32, 18, 0) )
  {
    v97[1] = 1;
    v95 = "Invalid record";
    v97[0] = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v95);
    return a1;
  }
  v2 = a2 + 32;
  v95 = v97;
  v96 = 0x4000000000LL;
  while ( 1 )
  {
    v3 = sub_14ED070(v2, 0);
    if ( (_DWORD)v3 == 1 )
      break;
    if ( (v3 & 0xFFFFFFFD) == 0 )
    {
      BYTE1(v93) = 1;
      v23 = "Malformed block";
LABEL_23:
      v91 = v23;
      LOBYTE(v93) = 3;
      sub_14EE4B0(a1, a2 + 8, (__int64)&v91);
      goto LABEL_24;
    }
    LODWORD(v96) = 0;
    v4 = sub_1510D70(v2, HIDWORD(v3), &v95, 0);
    if ( v4 == 1 )
    {
      v5 = 0;
LABEL_7:
      if ( (unsigned int)v96 <= 2 )
      {
        BYTE1(v93) = 1;
        v23 = "Invalid record";
        goto LABEL_23;
      }
      v6 = v96 - 1;
      v7 = *(_QWORD *)&v95[8 * (unsigned int)v96 - 8];
      LODWORD(v96) = v96 - 1;
      if ( v5 )
        v8 = *(_QWORD *)(*(_QWORD *)(a2 + 1368) + 8LL * (unsigned int)v7);
      else
        v8 = *(_QWORD *)(*(_QWORD *)(a2 + 552) + 24LL * (unsigned int)v7 + 16);
      v86 = v8;
      v91 = 0;
      v9 = (unsigned __int64 *)&v93;
      v92 = 1;
      do
      {
        *v9 = -8;
        v9 += 2;
      }
      while ( v9 != (unsigned __int64 *)&v95 );
      v10 = *(_QWORD *)(v86 + 8);
      if ( v10 )
      {
        v11 = 0;
        v12 = &v91;
        while ( 1 )
        {
          v18 = v92;
          v19 = v11 + 1;
          v20 = v92 & 1;
          if ( (int)v11 + 1 > v6 )
            goto LABEL_40;
          v21 = *(_QWORD *)&v95[8 * v11];
          if ( v20 )
          {
            v13 = 15;
            v14 = &v93;
          }
          else
          {
            v22 = v94;
            v14 = v93;
            if ( !v94 )
            {
              v25 = v92;
              ++v91;
              v16 = 0;
              v26 = ((unsigned int)v92 >> 1) + 1;
LABEL_32:
              v27 = 3 * v22;
              goto LABEL_33;
            }
            v13 = v94 - 1;
          }
          v15 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v16 = &v14[2 * v15];
          v17 = *v16;
          if ( v10 == *v16 )
            goto LABEL_16;
          v28 = 1;
          v29 = 0;
          while ( v17 != -8 )
          {
            if ( v17 == -16 && !v29 )
              v29 = v16;
            v15 = v13 & (v28 + v15);
            v16 = &v14[2 * v15];
            v17 = *v16;
            if ( v10 == *v16 )
              goto LABEL_16;
            ++v28;
          }
          v25 = v92;
          v27 = 48;
          v22 = 16;
          if ( v29 )
            v16 = v29;
          ++v91;
          v26 = ((unsigned int)v92 >> 1) + 1;
          if ( !v20 )
          {
            v22 = v94;
            goto LABEL_32;
          }
LABEL_33:
          if ( 4 * v26 >= v27 )
          {
            v81 = v12;
            sub_14FBFD0((__int64)v12, 2 * v22);
            v12 = v81;
            if ( (v92 & 1) != 0 )
            {
              v31 = 15;
              v30 = &v93;
            }
            else
            {
              v30 = v93;
              if ( !v94 )
                goto LABEL_153;
              v31 = v94 - 1;
            }
            v25 = v92;
            v32 = v31 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v16 = &v30[2 * v32];
            v33 = *v16;
            if ( v10 == *v16 )
              goto LABEL_35;
            v34 = 1;
            v35 = 0;
            while ( v33 != -8 )
            {
              if ( v33 == -16 && !v35 )
                v35 = v16;
              v32 = v31 & (v34 + v32);
              v16 = &v30[2 * v32];
              v33 = *v16;
              if ( v10 == *v16 )
                goto LABEL_56;
              ++v34;
            }
          }
          else
          {
            if ( v22 - HIDWORD(v92) - v26 > v22 >> 3 )
              goto LABEL_35;
            v82 = v12;
            sub_14FBFD0((__int64)v12, v22);
            v12 = v82;
            if ( (v92 & 1) != 0 )
            {
              v37 = 15;
              v36 = &v93;
            }
            else
            {
              v36 = v93;
              if ( !v94 )
              {
LABEL_153:
                LODWORD(v92) = (2 * ((unsigned int)v92 >> 1) + 2) | v92 & 1;
                BUG();
              }
              v37 = v94 - 1;
            }
            v25 = v92;
            v38 = v37 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v16 = &v36[2 * v38];
            v39 = *v16;
            if ( v10 == *v16 )
              goto LABEL_35;
            v40 = 1;
            v35 = 0;
            while ( v39 != -8 )
            {
              if ( v39 == -16 && !v35 )
                v35 = v16;
              v38 = v37 & (v40 + v38);
              v16 = &v36[2 * v38];
              v39 = *v16;
              if ( v10 == *v16 )
                goto LABEL_56;
              ++v40;
            }
          }
          if ( v35 )
            v16 = v35;
LABEL_56:
          v25 = v92;
LABEL_35:
          LODWORD(v92) = (2 * (v25 >> 1) + 2) | v25 & 1;
          if ( *v16 != -8 )
            --HIDWORD(v92);
          *v16 = v10;
          *((_DWORD *)v16 + 2) = 0;
LABEL_16:
          *((_DWORD *)v16 + 2) = v21;
          v10 = *(_QWORD *)(v10 + 8);
          v6 = v96;
          if ( !v10 )
          {
            v18 = v92;
            if ( (_DWORD)v96 != (unsigned int)v92 >> 1 )
              goto LABEL_40;
            if ( v19 > (unsigned int)v96 )
              goto LABEL_40;
            v41 = *(_QWORD *)(v86 + 8);
            if ( !v41 )
              goto LABEL_40;
            v42 = *(_QWORD *)(v41 + 8);
            if ( !v42 )
              goto LABEL_40;
            *(_QWORD *)(v41 + 8) = 0;
            v43 = *(_QWORD *)(v86 + 8);
            v44 = *(_QWORD *)(v42 + 8);
            v90[0] = v43;
            if ( v44 )
            {
              v83 = v44;
              v45 = 1;
              while ( 1 )
              {
                *(_QWORD *)(v42 + 8) = 0;
                v46 = v90;
                v47 = 0;
                v48 = v45;
                while ( 1 )
                {
                  v49 = *v46;
                  if ( !*v46 )
                    break;
                  v50 = &v89;
                  while ( 1 )
                  {
                    if ( !v42 )
                    {
LABEL_86:
                      *v50 = v49;
                      goto LABEL_87;
                    }
                    while ( 1 )
                    {
                      if ( (v92 & 1) != 0 )
                      {
                        v51 = 15;
                        v52 = &v93;
                      }
                      else
                      {
                        v52 = v93;
                        if ( !v94 )
                          break;
                        v51 = v94 - 1;
                      }
                      v53 = v51 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
                      v54 = &v52[2 * v53];
                      v55 = *v54;
                      if ( *v54 == v42 )
                      {
LABEL_81:
                        v56 = *((_DWORD *)v54 + 2);
                      }
                      else
                      {
                        v62 = 1;
                        while ( v55 != -8 )
                        {
                          v53 = v51 & (v62 + v53);
                          v80 = v62 + 1;
                          v54 = &v52[2 * v53];
                          v55 = *v54;
                          if ( *v54 == v42 )
                            goto LABEL_81;
                          v62 = v80;
                        }
                        v56 = 0;
                      }
                      if ( (v92 & 1) == 0 && !v94 )
                        break;
                      v57 = v51 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
                      v58 = &v52[2 * v57];
                      v59 = *v58;
                      if ( *v58 != v49 )
                      {
                        v60 = 1;
                        if ( v59 == -8 )
                          break;
                        while ( 1 )
                        {
                          v57 = v51 & (v60 + v57);
                          v79 = v60 + 1;
                          v58 = &v52[2 * v57];
                          v61 = *v58;
                          if ( *v58 == v49 )
                            break;
                          v60 = v79;
                          if ( v61 == -8 )
                            goto LABEL_96;
                        }
                      }
                      if ( *((_DWORD *)v58 + 2) <= v56 )
                        break;
                      *v50 = v42;
                      v50 = (__int64 *)(v42 + 8);
                      v42 = *(_QWORD *)(v42 + 8);
                      if ( !v42 )
                        goto LABEL_86;
                    }
LABEL_96:
                    *v50 = v49;
                    v50 = (__int64 *)(v49 + 8);
                    if ( !*(_QWORD *)(v49 + 8) )
                      break;
                    v49 = *(_QWORD *)(v49 + 8);
                  }
                  *(_QWORD *)(v49 + 8) = v42;
LABEL_87:
                  v47 = (unsigned int)(v47 + 1);
                  *v46 = 0;
                  v42 = v89;
                  ++v46;
                  if ( v48 == (_DWORD)v47 )
                  {
                    v45 = v48 + 1;
                    goto LABEL_89;
                  }
                }
                v45 = v48;
LABEL_89:
                v90[v47] = v42;
                v42 = v83;
                if ( !*(_QWORD *)(v83 + 8) )
                  break;
                v83 = *(_QWORD *)(v83 + 8);
              }
              v64 = v83;
              v65 = v45;
              v43 = v90[0];
              *(_QWORD *)(v86 + 8) = v83;
            }
            else
            {
              v64 = v42;
              *(_QWORD *)(v86 + 8) = v42;
              v65 = 1;
            }
            for ( i = 0; ; v43 = v90[i] )
            {
              v67 = &v89;
              if ( v43 )
              {
                while ( 1 )
                {
                  if ( !v64 )
                  {
LABEL_122:
                    *v67 = v43;
                    goto LABEL_123;
                  }
                  while ( 1 )
                  {
                    if ( (v92 & 1) != 0 )
                    {
                      v68 = 15;
                      v69 = &v93;
                    }
                    else
                    {
                      v69 = v93;
                      if ( !v94 )
                        break;
                      v68 = v94 - 1;
                    }
                    v70 = v68 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                    v71 = &v69[2 * v70];
                    v72 = *v71;
                    if ( *v71 == v64 )
                    {
LABEL_117:
                      v73 = *((_DWORD *)v71 + 2);
                    }
                    else
                    {
                      v77 = 1;
                      while ( v72 != -8 )
                      {
                        v70 = v68 & (v77 + v70);
                        v85 = v77 + 1;
                        v71 = &v69[2 * v70];
                        v72 = *v71;
                        if ( *v71 == v64 )
                          goto LABEL_117;
                        v77 = v85;
                      }
                      v73 = 0;
                    }
                    if ( (v92 & 1) == 0 && !v94 )
                      break;
                    v74 = v68 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                    v75 = &v69[2 * v74];
                    v76 = *v75;
                    if ( *v75 != v43 )
                    {
                      v78 = 1;
                      while ( v76 != -8 )
                      {
                        v74 = v68 & (v78 + v74);
                        v84 = v78 + 1;
                        v75 = &v69[2 * v74];
                        v76 = *v75;
                        if ( *v75 == v43 )
                          goto LABEL_120;
                        v78 = v84;
                      }
                      break;
                    }
LABEL_120:
                    if ( *((_DWORD *)v75 + 2) <= v73 )
                      break;
                    *v67 = v64;
                    v67 = (__int64 *)(v64 + 8);
                    v64 = *(_QWORD *)(v64 + 8);
                    if ( !v64 )
                      goto LABEL_122;
                  }
                  *v67 = v43;
                  v67 = (__int64 *)(v43 + 8);
                  if ( !*(_QWORD *)(v43 + 8) )
                    break;
                  v43 = *(_QWORD *)(v43 + 8);
                }
                *(_QWORD *)(v43 + 8) = v64;
LABEL_123:
                v64 = v89;
                *(_QWORD *)(v86 + 8) = v89;
              }
              if ( v65 <= (unsigned int)++i )
                break;
            }
            for ( j = v86 + 8; v64; v64 = *(_QWORD *)(v64 + 8) )
            {
              *(_QWORD *)(v64 + 16) = *(_QWORD *)(v64 + 16) & 3LL | j;
              j = v64 + 8;
            }
            break;
          }
          v11 = v19;
        }
      }
      v18 = v92;
LABEL_40:
      if ( (v18 & 1) == 0 )
        j___libc_free_0(v93);
    }
    else
    {
      v5 = 1;
      if ( v4 == 2 )
        goto LABEL_7;
    }
  }
  *a1 = 1;
LABEL_24:
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
  return a1;
}
