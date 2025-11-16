// Function: sub_1CCEBE0
// Address: 0x1ccebe0
//
__int64 __fastcall sub_1CCEBE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  int v6; // r8d
  unsigned int v7; // r15d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // r9
  __int64 v17; // r12
  unsigned int v18; // eax
  int v19; // edx
  __int64 v20; // rdi
  _BYTE *v21; // rdx
  int v22; // eax
  _BYTE *v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // r11
  __int64 v28; // rbx
  int v29; // r8d
  unsigned int v30; // edx
  _QWORD *v31; // rcx
  __int64 v32; // rax
  __int64 v33; // r12
  char v34; // al
  unsigned int v35; // r12d
  __int64 v37; // r9
  int v38; // r10d
  int v39; // r10d
  _QWORD *v40; // r9
  int v41; // edx
  __int64 v42; // rax
  int v43; // r10d
  __int64 *v44; // rdi
  unsigned int v45; // r15d
  __int64 v46; // rcx
  int v47; // eax
  unsigned int v48; // eax
  __int64 v49; // r14
  int v50; // edi
  _QWORD *v51; // rsi
  unsigned int v52; // r14d
  __int64 v53; // rcx
  int v54; // eax
  int v55; // esi
  __int64 *v56; // rcx
  int v57; // r10d
  __int64 v58; // rax
  unsigned int v59; // [rsp+10h] [rbp-E0h]
  __int64 v60; // [rsp+10h] [rbp-E0h]
  __int64 v61; // [rsp+10h] [rbp-E0h]
  __int64 v62; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v64; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v65; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v66; // [rsp+58h] [rbp-98h]
  __int64 v67; // [rsp+60h] [rbp-90h]
  __int64 v68; // [rsp+68h] [rbp-88h]
  _BYTE *v69; // [rsp+70h] [rbp-80h] BYREF
  __int64 v70; // [rsp+78h] [rbp-78h]
  _BYTE v71[112]; // [rsp+80h] [rbp-70h] BYREF

  v3 = a1 + 24;
  v4 = *(_QWORD *)(a1 + 32);
  v69 = v71;
  v70 = 0x800000000LL;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  if ( v4 == a1 + 24 )
  {
    v21 = v71;
    v22 = 0;
    goto LABEL_22;
  }
  do
  {
    v12 = v4 - 56;
    if ( !v4 )
      v12 = 0;
    if ( !sub_15E4F60(v12) || *(_QWORD *)(v12 + 8) && (*(_BYTE *)(v12 + 33) & 0x20) == 0 )
    {
      v13 = sub_1649960(v12);
      v15 = sub_16321A0(a2, (__int64)v13, v14);
      v17 = v15;
      if ( v15 )
      {
        if ( (_DWORD)v68 )
        {
          v6 = v68 - 1;
          v7 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
          v8 = (v68 - 1) & v7;
          v9 = (__int64 *)(v66 + 8LL * v8);
          v10 = *v9;
          if ( v17 == *v9 )
          {
LABEL_4:
            v11 = (unsigned int)v70;
            if ( (unsigned int)v70 < HIDWORD(v70) )
            {
LABEL_5:
              *(_QWORD *)&v69[8 * v11] = v17;
              LODWORD(v70) = v70 + 1;
              goto LABEL_6;
            }
LABEL_20:
            sub_16CD150((__int64)&v69, v71, 0, 8, v6, (int)v16);
            v11 = (unsigned int)v70;
            goto LABEL_5;
          }
          v43 = 1;
          v16 = 0;
          while ( v10 != -8 )
          {
            if ( v16 || v10 != -16 )
              v9 = v16;
            LODWORD(v16) = v43 + 1;
            v8 = v6 & (v43 + v8);
            v10 = *(_QWORD *)(v66 + 8LL * v8);
            if ( v17 == v10 )
              goto LABEL_4;
            ++v43;
            v16 = v9;
            v9 = (__int64 *)(v66 + 8LL * v8);
          }
          if ( !v16 )
            v16 = v9;
          ++v65;
          v19 = v67 + 1;
          if ( 4 * ((int)v67 + 1) < (unsigned int)(3 * v68) )
          {
            if ( (int)v68 - HIDWORD(v67) - v19 <= (unsigned int)v68 >> 3 )
            {
              sub_1CAE960((__int64)&v65, v68);
              if ( !(_DWORD)v68 )
              {
LABEL_125:
                LODWORD(v67) = v67 + 1;
                BUG();
              }
              v6 = v66;
              v44 = 0;
              v45 = (v68 - 1) & v7;
              v16 = (__int64 *)(v66 + 8LL * v45);
              v46 = *v16;
              v19 = v67 + 1;
              v47 = 1;
              if ( v17 != *v16 )
              {
                while ( v46 != -8 )
                {
                  if ( v46 == -16 && !v44 )
                    v44 = v16;
                  v45 = (v68 - 1) & (v47 + v45);
                  v16 = (__int64 *)(v66 + 8LL * v45);
                  v46 = *v16;
                  if ( v17 == *v16 )
                    goto LABEL_17;
                  ++v47;
                }
                if ( v44 )
                  v16 = v44;
              }
            }
LABEL_17:
            LODWORD(v67) = v19;
            if ( *v16 != -8 )
              --HIDWORD(v67);
            *v16 = v17;
            v11 = (unsigned int)v70;
            if ( (unsigned int)v70 < HIDWORD(v70) )
              goto LABEL_5;
            goto LABEL_20;
          }
        }
        else
        {
          ++v65;
        }
        sub_1CAE960((__int64)&v65, 2 * v68);
        if ( !(_DWORD)v68 )
          goto LABEL_125;
        v6 = v66;
        v18 = (v68 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v16 = (__int64 *)(v66 + 8LL * v18);
        v19 = v67 + 1;
        v20 = *v16;
        if ( v17 != *v16 )
        {
          v55 = 1;
          v56 = 0;
          while ( v20 != -8 )
          {
            if ( v20 == -16 && !v56 )
              v56 = v16;
            v18 = (v68 - 1) & (v55 + v18);
            v16 = (__int64 *)(v66 + 8LL * v18);
            v20 = *v16;
            if ( v17 == *v16 )
              goto LABEL_17;
            ++v55;
          }
          if ( v56 )
            v16 = v56;
        }
        goto LABEL_17;
      }
    }
LABEL_6:
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v3 != v4 );
  v21 = v69;
  v22 = v70;
LABEL_22:
  v23 = &v21[8 * v22];
  while ( 1 )
  {
    if ( !v22 )
    {
      v35 = 0;
      goto LABEL_41;
    }
    v24 = *((_QWORD *)v23 - 1);
    LODWORD(v70) = --v22;
    if ( !v24 )
      break;
    v23 -= 8;
    if ( (*(_BYTE *)(v24 + 34) & 0x40) != 0 )
    {
      sub_15E4B20((__int64)&v64, v24);
      if ( (v64 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v64 = v64 & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_16BCAE0(&v64, v24, v25);
      }
      v26 = *(_QWORD *)(v24 + 80);
      v27 = v24 + 72;
      if ( v26 != v24 + 72 )
      {
        while ( 1 )
        {
          if ( !v26 )
            BUG();
          v28 = *(_QWORD *)(v26 + 24);
          if ( v28 != v26 + 16 )
            break;
LABEL_44:
          v26 = *(_QWORD *)(v26 + 8);
          if ( v27 == v26 )
            goto LABEL_45;
        }
        while ( 2 )
        {
          if ( !v28 )
            BUG();
          if ( *(_BYTE *)(v28 - 8) == 78 )
          {
            v33 = *(_QWORD *)(v28 - 48);
            v34 = *(_BYTE *)(v33 + 16);
            if ( v34 != 20 )
            {
              if ( v34 )
                goto LABEL_40;
              if ( (*(_BYTE *)(v33 + 33) & 0x20) == 0 )
              {
                if ( (_DWORD)v68 )
                {
                  v29 = v68 - 1;
                  v30 = (v68 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
                  v31 = (_QWORD *)(v66 + 8LL * v30);
                  v32 = *v31;
                  if ( v33 == *v31 )
                    goto LABEL_35;
                  v59 = (v68 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
                  v37 = *v31;
                  v38 = 1;
                  while ( v37 != -8 )
                  {
                    v59 = v29 & (v59 + v38);
                    v37 = *(_QWORD *)(v66 + 8LL * v59);
                    if ( v33 == v37 )
                      goto LABEL_35;
                    ++v38;
                  }
                  v39 = 1;
                  v40 = 0;
                  while ( v32 != -8 )
                  {
                    if ( v40 || v32 != -16 )
                      v31 = v40;
                    LODWORD(v40) = v39 + 1;
                    v30 = v29 & (v39 + v30);
                    v32 = *(_QWORD *)(v66 + 8LL * v30);
                    if ( v33 == v32 )
                      goto LABEL_59;
                    ++v39;
                    v40 = v31;
                    v31 = (_QWORD *)(v66 + 8LL * v30);
                  }
                  if ( !v40 )
                    v40 = v31;
                  ++v65;
                  v41 = v67 + 1;
                  if ( 4 * ((int)v67 + 1) < (unsigned int)(3 * v68) )
                  {
                    if ( (int)v68 - HIDWORD(v67) - v41 <= (unsigned int)v68 >> 3 )
                    {
                      v62 = v27;
                      sub_1CAE960((__int64)&v65, v68);
                      if ( !(_DWORD)v68 )
                        goto LABEL_125;
                      v29 = v66;
                      v51 = 0;
                      v27 = v62;
                      v52 = (v68 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
                      v40 = (_QWORD *)(v66 + 8LL * v52);
                      v53 = *v40;
                      v41 = v67 + 1;
                      v54 = 1;
                      if ( *v40 != v33 )
                      {
                        while ( v53 != -8 )
                        {
                          if ( v53 == -16 && !v51 )
                            v51 = v40;
                          v57 = v54 + 1;
                          v58 = ((_DWORD)v68 - 1) & (v52 + v54);
                          v40 = (_QWORD *)(v66 + 8 * v58);
                          v52 = v58;
                          v53 = *v40;
                          if ( v33 == *v40 )
                            goto LABEL_56;
                          v54 = v57;
                        }
                        goto LABEL_91;
                      }
                    }
                    goto LABEL_56;
                  }
                }
                else
                {
                  ++v65;
                }
                v61 = v27;
                sub_1CAE960((__int64)&v65, 2 * v68);
                if ( !(_DWORD)v68 )
                  goto LABEL_125;
                v29 = v66;
                v27 = v61;
                v48 = (v68 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
                v40 = (_QWORD *)(v66 + 8LL * v48);
                v41 = v67 + 1;
                v49 = *v40;
                if ( *v40 != v33 )
                {
                  v50 = 1;
                  v51 = 0;
                  while ( v49 != -8 )
                  {
                    if ( !v51 && v49 == -16 )
                      v51 = v40;
                    v48 = (v68 - 1) & (v48 + v50);
                    v40 = (_QWORD *)(v66 + 8LL * v48);
                    v49 = *v40;
                    if ( v33 == *v40 )
                      goto LABEL_56;
                    ++v50;
                  }
LABEL_91:
                  if ( v51 )
                    v40 = v51;
                }
LABEL_56:
                LODWORD(v67) = v41;
                if ( *v40 != -8 )
                  --HIDWORD(v67);
                *v40 = v33;
LABEL_59:
                v42 = (unsigned int)v70;
                if ( (unsigned int)v70 >= HIDWORD(v70) )
                {
                  v60 = v27;
                  sub_16CD150((__int64)&v69, v71, 0, 8, v29, (int)v40);
                  v42 = (unsigned int)v70;
                  v27 = v60;
                }
                *(_QWORD *)&v69[8 * v42] = v33;
                LODWORD(v70) = v70 + 1;
              }
            }
          }
LABEL_35:
          v28 = *(_QWORD *)(v28 + 8);
          if ( v26 + 16 == v28 )
            goto LABEL_44;
          continue;
        }
      }
LABEL_45:
      v21 = v69;
      v22 = v70;
      goto LABEL_22;
    }
  }
  if ( a3 )
    sub_2241130(a3, 0, *(_QWORD *)(a3 + 8), "Unknown function called.", 24);
LABEL_40:
  v35 = 1;
LABEL_41:
  j___libc_free_0(v66);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  return v35;
}
