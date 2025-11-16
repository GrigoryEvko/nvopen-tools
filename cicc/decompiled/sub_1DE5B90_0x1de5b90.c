// Function: sub_1DE5B90
// Address: 0x1de5b90
//
__int64 __fastcall sub_1DE5B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _BYTE *a7)
{
  __int64 v10; // rsi
  unsigned __int8 v11; // al
  __int64 *v12; // rdi
  __int64 *v13; // r14
  __int64 *v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r8
  int v17; // edi
  int v18; // r10d
  unsigned int v19; // esi
  __int64 v20; // r9
  __int64 v21; // r11
  __int64 *v22; // r12
  __int64 *v23; // rdx
  __int64 *v24; // r14
  __int64 *v25; // r10
  __int64 v26; // rdx
  int v27; // ecx
  int v28; // edi
  unsigned int v29; // eax
  __int64 v30; // rsi
  unsigned int v31; // esi
  __int64 v32; // r8
  unsigned int v33; // edi
  _QWORD *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  bool v37; // dl
  __int64 v38; // rbx
  int v39; // eax
  __int64 result; // rax
  int v41; // edi
  _QWORD *v42; // rdx
  int v43; // eax
  int v44; // eax
  int v45; // esi
  int v46; // esi
  __int64 v47; // r8
  unsigned int v48; // ecx
  __int64 v49; // rdi
  int v50; // r13d
  _QWORD *v51; // r9
  int v52; // ecx
  int v53; // ecx
  __int64 v54; // rdi
  unsigned int v55; // r13d
  int v56; // r8d
  __int64 v57; // rsi
  __int64 *v58; // [rsp+0h] [rbp-110h]
  __int64 *v59; // [rsp+0h] [rbp-110h]
  __int64 *v60; // [rsp+0h] [rbp-110h]
  int v61; // [rsp+8h] [rbp-108h]
  __int64 v62; // [rsp+8h] [rbp-108h]
  __int64 v63; // [rsp+8h] [rbp-108h]
  __int64 *v64; // [rsp+20h] [rbp-F0h]
  __int64 v65; // [rsp+28h] [rbp-E8h]
  char v67; // [rsp+3Fh] [rbp-D1h]
  unsigned __int8 v69; // [rsp+40h] [rbp-D0h]
  __int64 v70; // [rsp+48h] [rbp-C8h] BYREF
  unsigned __int8 v71; // [rsp+57h] [rbp-B9h] BYREF
  __int64 v72; // [rsp+58h] [rbp-B8h] BYREF
  _QWORD v73[2]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v74[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v75; // [rsp+90h] [rbp-80h] BYREF
  __int64 v76; // [rsp+98h] [rbp-78h]
  _BYTE v77[112]; // [rsp+A0h] [rbp-70h] BYREF

  *a7 = 0;
  v70 = a5;
  v10 = (unsigned __int8)sub_1F34100(a2);
  if ( (unsigned int)((__int64)(*(_QWORD *)(a2 + 96) - *(_QWORD *)(a2 + 88)) >> 3) == 1 )
    return 0;
  v67 = sub_1F34340(a1 + 616, v10, a2);
  if ( !v67 )
    return 0;
  v74[2] = a6;
  v74[0] = &v71;
  v74[3] = &v70;
  v73[0] = sub_1DE90D0;
  v73[1] = v74;
  v75 = (__int64 *)v77;
  v71 = 0;
  v74[1] = a1;
  v76 = 0x800000000LL;
  v11 = sub_1F34100(a2);
  sub_1F389B0(a1 + 616, v11, a2, a3, &v75, v73);
  v12 = v75;
  *a7 = 0;
  v64 = &v12[(unsigned int)v76];
  if ( v64 != v12 )
  {
    v13 = v12;
    v65 = a1 + 888;
    while ( 1 )
    {
      while ( 1 )
      {
        v72 = *v13;
        v14 = sub_1DE4FA0(v65, &v72);
        if ( v72 != a3 )
          break;
        ++v13;
        *a7 = 1;
        if ( v64 == v13 )
          goto LABEL_29;
      }
      v15 = v70;
      if ( !v70 )
        goto LABEL_10;
      if ( (*(_BYTE *)(v70 + 8) & 1) != 0 )
      {
        v16 = v70 + 16;
        v17 = 15;
      }
      else
      {
        v41 = *(_DWORD *)(v70 + 24);
        v16 = *(_QWORD *)(v70 + 16);
        if ( !v41 )
          goto LABEL_28;
        v17 = v41 - 1;
      }
      v18 = 1;
      v19 = v17 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      v20 = *(_QWORD *)(v16 + 8LL * v19);
      if ( v72 != v20 )
      {
        while ( v20 != -8 )
        {
          v19 = v17 & (v18 + v19);
          v20 = *(_QWORD *)(v16 + 8LL * v19);
          if ( v72 == v20 )
            goto LABEL_10;
          ++v18;
        }
        if ( v64 == ++v13 )
        {
LABEL_29:
          v12 = v75;
          break;
        }
      }
      else
      {
LABEL_10:
        v21 = v14[1];
        if ( v21 != a4 )
        {
          v22 = *(__int64 **)(v72 + 88);
          if ( *(__int64 **)(v72 + 96) != v22 )
          {
            v23 = v13;
            v24 = *(__int64 **)(v72 + 96);
            v25 = v23;
            while ( 2 )
            {
              v38 = *v22;
              if ( !v15 )
                goto LABEL_15;
              if ( (*(_BYTE *)(v15 + 8) & 1) != 0 )
              {
                v26 = v15 + 16;
                v27 = 15;
              }
              else
              {
                v26 = *(_QWORD *)(v15 + 16);
                v39 = *(_DWORD *)(v15 + 24);
                v27 = v39 - 1;
                if ( !v39 )
                  goto LABEL_21;
              }
              v28 = 1;
              v29 = v27 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
              v30 = *(_QWORD *)(v26 + 8LL * v29);
              if ( v38 != v30 )
              {
                while ( v30 != -8 )
                {
                  v29 = v27 & (v28 + v29);
                  v30 = *(_QWORD *)(v26 + 8LL * v29);
                  if ( v38 == v30 )
                    goto LABEL_15;
                  ++v28;
                }
              }
              else
              {
LABEL_15:
                v31 = *(_DWORD *)(a1 + 912);
                if ( v31 )
                {
                  v32 = *(_QWORD *)(a1 + 896);
                  v33 = (v31 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                  v34 = (_QWORD *)(v32 + 16LL * v33);
                  v35 = *v34;
                  if ( v38 == *v34 )
                  {
                    v36 = v34[1];
                    v37 = a4 != v36;
                    goto LABEL_18;
                  }
                  v61 = 1;
                  v42 = 0;
                  while ( v35 != -8 )
                  {
                    if ( v35 != -16 || v42 )
                      v34 = v42;
                    v33 = (v31 - 1) & (v61 + v33);
                    v60 = (__int64 *)(v32 + 16LL * v33);
                    v35 = *v60;
                    if ( v38 == *v60 )
                    {
                      v36 = v60[1];
                      v37 = a4 != v36;
                      goto LABEL_18;
                    }
                    ++v61;
                    v42 = v34;
                    v34 = (_QWORD *)(v32 + 16LL * v33);
                  }
                  if ( !v42 )
                    v42 = v34;
                  v43 = *(_DWORD *)(a1 + 904);
                  ++*(_QWORD *)(a1 + 888);
                  v44 = v43 + 1;
                  if ( 4 * v44 < 3 * v31 )
                  {
                    if ( v31 - *(_DWORD *)(a1 + 908) - v44 <= v31 >> 3 )
                    {
                      v59 = v25;
                      v63 = v21;
                      sub_1DE4DF0(v65, v31);
                      v52 = *(_DWORD *)(a1 + 912);
                      if ( !v52 )
                      {
LABEL_83:
                        ++*(_DWORD *)(a1 + 904);
                        BUG();
                      }
                      v53 = v52 - 1;
                      v54 = *(_QWORD *)(a1 + 896);
                      v51 = 0;
                      v55 = v53 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                      v21 = v63;
                      v25 = v59;
                      v56 = 1;
                      v44 = *(_DWORD *)(a1 + 904) + 1;
                      v42 = (_QWORD *)(v54 + 16LL * v55);
                      v57 = *v42;
                      if ( v38 != *v42 )
                      {
                        while ( v57 != -8 )
                        {
                          if ( v57 == -16 && !v51 )
                            v51 = v42;
                          v55 = v53 & (v56 + v55);
                          v42 = (_QWORD *)(v54 + 16LL * v55);
                          v57 = *v42;
                          if ( v38 == *v42 )
                            goto LABEL_47;
                          ++v56;
                        }
                        goto LABEL_55;
                      }
                    }
                    goto LABEL_47;
                  }
                }
                else
                {
                  ++*(_QWORD *)(a1 + 888);
                }
                v58 = v25;
                v62 = v21;
                sub_1DE4DF0(v65, 2 * v31);
                v45 = *(_DWORD *)(a1 + 912);
                if ( !v45 )
                  goto LABEL_83;
                v46 = v45 - 1;
                v47 = *(_QWORD *)(a1 + 896);
                v21 = v62;
                v25 = v58;
                v48 = v46 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                v44 = *(_DWORD *)(a1 + 904) + 1;
                v42 = (_QWORD *)(v47 + 16LL * v48);
                v49 = *v42;
                if ( v38 != *v42 )
                {
                  v50 = 1;
                  v51 = 0;
                  while ( v49 != -8 )
                  {
                    if ( v49 == -16 && !v51 )
                      v51 = v42;
                    v48 = v46 & (v50 + v48);
                    v42 = (_QWORD *)(v47 + 16LL * v48);
                    v49 = *v42;
                    if ( v38 == *v42 )
                      goto LABEL_47;
                    ++v50;
                  }
LABEL_55:
                  if ( v51 )
                    v42 = v51;
                }
LABEL_47:
                *(_DWORD *)(a1 + 904) = v44;
                if ( *v42 != -8 )
                  --*(_DWORD *)(a1 + 908);
                *v42 = v38;
                v36 = 0;
                v42[1] = 0;
                v37 = v67;
LABEL_18:
                if ( v21 != v36 && v37 )
                  ++*(_DWORD *)(v36 + 56);
              }
LABEL_21:
              if ( v24 == ++v22 )
              {
                v13 = v25;
                break;
              }
              v15 = v70;
              continue;
            }
          }
        }
LABEL_28:
        if ( v64 == ++v13 )
          goto LABEL_29;
      }
    }
  }
  result = v71;
  if ( v12 != (__int64 *)v77 )
  {
    v69 = v71;
    _libc_free((unsigned __int64)v12);
    return v69;
  }
  return result;
}
