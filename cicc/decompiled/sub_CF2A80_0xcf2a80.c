// Function: sub_CF2A80
// Address: 0xcf2a80
//
__int64 __fastcall sub_CF2A80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v6; // r8
  unsigned int v7; // r15d
  unsigned int v8; // ecx
  _QWORD *v9; // rax
  _BYTE *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // r12
  const char *v14; // rax
  unsigned __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // r9
  _BYTE *v18; // r12
  unsigned int v19; // eax
  int v20; // edx
  __int64 v21; // rdi
  _BYTE *v22; // rdx
  int v23; // eax
  _BYTE *v24; // rdx
  __int64 v25; // r15
  __int64 v26; // r11
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // r12
  unsigned int v31; // r12d
  __int64 v32; // rsi
  __int64 v34; // r8
  unsigned int v35; // ecx
  __int64 *v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // r9
  int v39; // r10d
  __int64 v40; // r9
  int v41; // edx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  int v44; // r10d
  __int64 v45; // rdi
  unsigned int v46; // r15d
  __int64 v47; // rcx
  int v48; // eax
  unsigned int v49; // eax
  __int64 v50; // r14
  int v51; // edi
  __int64 v52; // rsi
  unsigned int v53; // r14d
  __int64 v54; // rcx
  int v55; // eax
  unsigned int v56; // r10d
  int v57; // esi
  __int64 v58; // rcx
  int v59; // r10d
  __int64 v60; // rax
  int v61; // [rsp+Ch] [rbp-E4h]
  int v62; // [rsp+10h] [rbp-E0h]
  __int64 v63; // [rsp+10h] [rbp-E0h]
  __int64 v64; // [rsp+10h] [rbp-E0h]
  __int64 v65; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v67; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v68; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v69; // [rsp+58h] [rbp-98h]
  __int64 v70; // [rsp+60h] [rbp-90h]
  __int64 v71; // [rsp+68h] [rbp-88h]
  _BYTE *v72; // [rsp+70h] [rbp-80h] BYREF
  __int64 v73; // [rsp+78h] [rbp-78h]
  _BYTE v74[112]; // [rsp+80h] [rbp-70h] BYREF

  v3 = a1 + 24;
  v4 = *(_QWORD *)(a1 + 32);
  v72 = v74;
  v73 = 0x800000000LL;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  if ( v4 == a1 + 24 )
  {
    v22 = v74;
    v23 = 0;
    goto LABEL_22;
  }
  do
  {
    v13 = v4 - 56;
    if ( !v4 )
      v13 = 0;
    if ( !sub_B2FC80(v13) || *(_QWORD *)(v13 + 16) && (*(_BYTE *)(v13 + 33) & 0x20) == 0 )
    {
      v14 = sub_BD5D20(v13);
      v16 = sub_BA8CB0(a2, (__int64)v14, v15);
      v18 = v16;
      if ( v16 )
      {
        if ( (_DWORD)v71 )
        {
          v6 = (unsigned int)(v71 - 1);
          v7 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
          v8 = v6 & v7;
          v9 = (_QWORD *)(v69 + 8LL * ((unsigned int)v6 & v7));
          v10 = (_BYTE *)*v9;
          if ( v18 == (_BYTE *)*v9 )
          {
LABEL_4:
            v11 = (unsigned int)v73;
            v12 = (unsigned int)v73 + 1LL;
            if ( v12 <= HIDWORD(v73) )
            {
LABEL_5:
              *(_QWORD *)&v72[8 * v11] = v18;
              LODWORD(v73) = v73 + 1;
              goto LABEL_6;
            }
LABEL_20:
            sub_C8D5F0((__int64)&v72, v74, v12, 8u, v6, v17);
            v11 = (unsigned int)v73;
            goto LABEL_5;
          }
          v44 = 1;
          v17 = 0;
          while ( v10 != (_BYTE *)-4096LL )
          {
            if ( v10 != (_BYTE *)-8192LL || v17 )
              v9 = (_QWORD *)v17;
            v17 = (unsigned int)(v44 + 1);
            v8 = v6 & (v44 + v8);
            v10 = *(_BYTE **)(v69 + 8LL * v8);
            if ( v18 == v10 )
              goto LABEL_4;
            ++v44;
            v17 = (__int64)v9;
            v9 = (_QWORD *)(v69 + 8LL * v8);
          }
          if ( !v17 )
            v17 = (__int64)v9;
          ++v68;
          v20 = v70 + 1;
          if ( 4 * ((int)v70 + 1) < (unsigned int)(3 * v71) )
          {
            if ( (int)v71 - HIDWORD(v70) - v20 <= (unsigned int)v71 >> 3 )
            {
              sub_A35F10((__int64)&v68, v71);
              if ( !(_DWORD)v71 )
              {
LABEL_129:
                LODWORD(v70) = v70 + 1;
                BUG();
              }
              v6 = v69;
              v45 = 0;
              v46 = (v71 - 1) & v7;
              v17 = v69 + 8LL * v46;
              v47 = *(_QWORD *)v17;
              v20 = v70 + 1;
              v48 = 1;
              if ( v18 != *(_BYTE **)v17 )
              {
                while ( v47 != -4096 )
                {
                  if ( v47 == -8192 && !v45 )
                    v45 = v17;
                  v46 = (v71 - 1) & (v48 + v46);
                  v17 = v69 + 8LL * v46;
                  v47 = *(_QWORD *)v17;
                  if ( v18 == *(_BYTE **)v17 )
                    goto LABEL_17;
                  ++v48;
                }
                if ( v45 )
                  v17 = v45;
              }
            }
LABEL_17:
            LODWORD(v70) = v20;
            if ( *(_QWORD *)v17 != -4096 )
              --HIDWORD(v70);
            *(_QWORD *)v17 = v18;
            v11 = (unsigned int)v73;
            v12 = (unsigned int)v73 + 1LL;
            if ( v12 <= HIDWORD(v73) )
              goto LABEL_5;
            goto LABEL_20;
          }
        }
        else
        {
          ++v68;
        }
        sub_A35F10((__int64)&v68, 2 * v71);
        if ( !(_DWORD)v71 )
          goto LABEL_129;
        v6 = v69;
        v19 = (v71 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v17 = v69 + 8LL * v19;
        v20 = v70 + 1;
        v21 = *(_QWORD *)v17;
        if ( v18 != *(_BYTE **)v17 )
        {
          v57 = 1;
          v58 = 0;
          while ( v21 != -4096 )
          {
            if ( !v58 && v21 == -8192 )
              v58 = v17;
            v19 = (v71 - 1) & (v57 + v19);
            v17 = v69 + 8LL * v19;
            v21 = *(_QWORD *)v17;
            if ( v18 == *(_BYTE **)v17 )
              goto LABEL_17;
            ++v57;
          }
          if ( v58 )
            v17 = v58;
        }
        goto LABEL_17;
      }
    }
LABEL_6:
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v3 != v4 );
  v22 = v72;
  v23 = v73;
LABEL_22:
  v24 = &v22[8 * v23];
  while ( 1 )
  {
    if ( !v23 )
    {
      v31 = 0;
      goto LABEL_37;
    }
    v25 = *((_QWORD *)v24 - 1);
    LODWORD(v73) = --v23;
    if ( !v25 )
      break;
    v24 -= 8;
    if ( (*(_BYTE *)(v25 + 35) & 8) != 0 )
    {
      sub_B2F620((__int64)&v67, v25);
      if ( (v67 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v67 = v67 & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_C63C30(&v67, v25);
      }
      v26 = *(_QWORD *)(v25 + 80);
      v27 = v25 + 72;
      if ( v26 != v27 )
      {
        while ( 1 )
        {
          if ( !v26 )
            BUG();
          v28 = *(_QWORD *)(v26 + 32);
          v29 = v26 + 24;
          if ( v28 != v26 + 24 )
            break;
LABEL_44:
          v26 = *(_QWORD *)(v26 + 8);
          if ( v27 == v26 )
            goto LABEL_45;
        }
        while ( 2 )
        {
          while ( 2 )
          {
            if ( !v28 )
              BUG();
            if ( *(_BYTE *)(v28 - 24) != 85 )
              goto LABEL_43;
            v30 = *(_QWORD *)(v28 - 56);
            if ( *(_BYTE *)v30 == 25 )
              goto LABEL_43;
            if ( *(_BYTE *)v30 || *(_QWORD *)(v30 + 24) != *(_QWORD *)(v28 + 56) )
              goto LABEL_36;
            if ( (*(_BYTE *)(v30 + 33) & 0x20) != 0 )
            {
LABEL_43:
              v28 = *(_QWORD *)(v28 + 8);
              if ( v29 == v28 )
                goto LABEL_44;
              continue;
            }
            break;
          }
          if ( (_DWORD)v71 )
          {
            v34 = (unsigned int)(v71 - 1);
            v35 = v34 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v36 = (__int64 *)(v69 + 8LL * v35);
            v37 = *v36;
            if ( v30 == *v36 )
              goto LABEL_43;
            v61 = 1;
            v38 = *v36;
            v62 = v34 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            while ( v38 != -4096 )
            {
              v56 = v34 & (v62 + v61++);
              v38 = *(_QWORD *)(v69 + 8LL * v56);
              v62 = v56;
              if ( v30 == v38 )
                goto LABEL_43;
            }
            v39 = 1;
            v40 = 0;
            while ( v37 != -4096 )
            {
              if ( v40 || v37 != -8192 )
                v36 = (__int64 *)v40;
              v40 = (unsigned int)(v39 + 1);
              v35 = v34 & (v39 + v35);
              v37 = *(_QWORD *)(v69 + 8LL * v35);
              if ( v30 == v37 )
                goto LABEL_57;
              ++v39;
              v40 = (__int64)v36;
              v36 = (__int64 *)(v69 + 8LL * v35);
            }
            if ( !v40 )
              v40 = (__int64)v36;
            ++v68;
            v41 = v70 + 1;
            if ( 4 * ((int)v70 + 1) < (unsigned int)(3 * v71) )
            {
              if ( (int)v71 - HIDWORD(v70) - v41 <= (unsigned int)v71 >> 3 )
              {
                v65 = v26;
                sub_A35F10((__int64)&v68, v71);
                if ( !(_DWORD)v71 )
                  goto LABEL_129;
                v34 = 0;
                v26 = v65;
                v53 = (v71 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v40 = v69 + 8LL * v53;
                v54 = *(_QWORD *)v40;
                v41 = v70 + 1;
                v55 = 1;
                if ( *(_QWORD *)v40 != v30 )
                {
                  while ( v54 != -4096 )
                  {
                    if ( v54 == -8192 && !v34 )
                      v34 = v40;
                    v59 = v55 + 1;
                    v60 = ((_DWORD)v71 - 1) & (v53 + v55);
                    v40 = v69 + 8 * v60;
                    v53 = v60;
                    v54 = *(_QWORD *)v40;
                    if ( v30 == *(_QWORD *)v40 )
                      goto LABEL_54;
                    v55 = v59;
                  }
                  if ( v34 )
                    v40 = v34;
                }
              }
LABEL_54:
              LODWORD(v70) = v41;
              if ( *(_QWORD *)v40 != -4096 )
                --HIDWORD(v70);
              *(_QWORD *)v40 = v30;
LABEL_57:
              v42 = (unsigned int)v73;
              v43 = (unsigned int)v73 + 1LL;
              if ( v43 > HIDWORD(v73) )
              {
                v63 = v26;
                sub_C8D5F0((__int64)&v72, v74, v43, 8u, v34, v40);
                v42 = (unsigned int)v73;
                v26 = v63;
              }
              *(_QWORD *)&v72[8 * v42] = v30;
              LODWORD(v73) = v73 + 1;
              v28 = *(_QWORD *)(v28 + 8);
              if ( v29 == v28 )
                goto LABEL_44;
              continue;
            }
          }
          else
          {
            ++v68;
          }
          break;
        }
        v64 = v26;
        sub_A35F10((__int64)&v68, 2 * v71);
        if ( !(_DWORD)v71 )
          goto LABEL_129;
        v34 = v69;
        v26 = v64;
        v49 = (v71 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v40 = v69 + 8LL * v49;
        v41 = v70 + 1;
        v50 = *(_QWORD *)v40;
        if ( *(_QWORD *)v40 != v30 )
        {
          v51 = 1;
          v52 = 0;
          while ( v50 != -4096 )
          {
            if ( v50 == -8192 && !v52 )
              v52 = v40;
            v49 = (v71 - 1) & (v49 + v51);
            v40 = v69 + 8LL * v49;
            v50 = *(_QWORD *)v40;
            if ( v30 == *(_QWORD *)v40 )
              goto LABEL_54;
            ++v51;
          }
          if ( v52 )
            v40 = v52;
        }
        goto LABEL_54;
      }
LABEL_45:
      v22 = v72;
      v23 = v73;
      goto LABEL_22;
    }
  }
  if ( a3 )
    sub_2241130(a3, 0, *(_QWORD *)(a3 + 8), "Unknown function called.", 24);
LABEL_36:
  v31 = 1;
LABEL_37:
  v32 = 8LL * (unsigned int)v71;
  sub_C7D6A0(v69, v32, 8);
  if ( v72 != v74 )
    _libc_free(v72, v32);
  return v31;
}
