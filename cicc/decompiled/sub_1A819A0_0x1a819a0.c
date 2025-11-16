// Function: sub_1A819A0
// Address: 0x1a819a0
//
__int64 __fastcall sub_1A819A0(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // r15
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r14
  __int64 v17; // rdi
  unsigned __int8 v18; // al
  __int64 v19; // r13
  unsigned __int64 v20; // r12
  __int64 v21; // rax
  __int64 *v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rsi
  __int64 v26; // r13
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rbx
  __int64 v30; // r12
  _QWORD *v31; // r12
  _QWORD *v32; // rax
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // rdx
  unsigned int v38; // ebx
  __int64 v39; // r15
  __int64 v40; // r9
  char *v41; // rax
  char *v42; // r9
  __int64 v43; // r9
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // r8
  _BYTE *v46; // rcx
  int v47; // edi
  _BYTE *v48; // r10
  __int64 v49; // rsi
  int v50; // r13d
  unsigned __int8 v51; // al
  double v52; // xmm4_8
  double v53; // xmm5_8
  __int64 v54; // [rsp+0h] [rbp-110h]
  __int64 v55; // [rsp+8h] [rbp-108h]
  char *v56; // [rsp+20h] [rbp-F0h]
  __int64 v57; // [rsp+28h] [rbp-E8h]
  __int64 v59; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v60; // [rsp+38h] [rbp-D8h]
  __int64 v62; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v63; // [rsp+48h] [rbp-C8h]
  __int64 v64; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v65; // [rsp+60h] [rbp-B0h]
  unsigned __int64 v66; // [rsp+60h] [rbp-B0h]
  __int64 v67; // [rsp+68h] [rbp-A8h]
  unsigned __int8 v69; // [rsp+7Eh] [rbp-92h]
  char v70; // [rsp+7Fh] [rbp-91h]
  __int64 v71; // [rsp+88h] [rbp-88h] BYREF
  _BYTE *v72; // [rsp+90h] [rbp-80h] BYREF
  __int64 v73; // [rsp+98h] [rbp-78h]
  _BYTE v74[16]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE *v75; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v76; // [rsp+B8h] [rbp-58h]
  _BYTE v77[80]; // [rsp+C0h] [rbp-50h] BYREF

  v67 = *(_QWORD *)(a1 + 80);
  v64 = a1 + 72;
  v69 = 0;
  if ( v67 != a1 + 72 )
  {
    while ( 1 )
    {
      v12 = v67 - 24;
      v13 = v67 - 24;
      v67 = *(_QWORD *)(v67 + 8);
      v14 = sub_157ED60(v13) + 24;
      v15 = sub_157EBA0(v12);
      v16 = v15 + 24;
      if ( v14 != v15 + 24 )
        break;
LABEL_25:
      if ( v67 == v64 )
        return v69;
    }
    while ( 1 )
    {
      v17 = v14 - 24;
      if ( !v14 )
        v17 = 0;
      if ( v17 == v15 )
        goto LABEL_25;
      v18 = *(_BYTE *)(v17 + 16);
      v14 = *(_QWORD *)(v14 + 8);
      if ( v18 > 0x17u )
      {
        if ( v18 == 78 )
        {
          v19 = v17 | 4;
          v20 = v17 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v17 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_5;
          v21 = *(_QWORD *)(v17 - 24);
          if ( !*(_BYTE *)(v21 + 16) && (*(_BYTE *)(v21 + 33) & 0x20) != 0 )
            goto LABEL_5;
        }
        else
        {
          if ( v18 != 29 )
            goto LABEL_5;
          v19 = v17 & 0xFFFFFFFFFFFFFFFBLL;
          v20 = v17 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v17 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_5;
        }
        if ( !(unsigned __int8)sub_1AE9990(v17, a2) )
        {
          v22 = (__int64 *)(v20 - 72);
          if ( ((v19 >> 2) & 1) != 0 )
            v22 = (__int64 *)(v20 - 24);
          v70 = (v19 >> 2) & 1;
          if ( !*(_BYTE *)(*v22 + 16) && !sub_15E4F60(*v22) )
            break;
        }
      }
LABEL_5:
      if ( v16 == v14 )
        goto LABEL_25;
      v15 = sub_157EBA0(v12);
    }
    if ( ((v19 >> 2) & 1) != 0 )
      v70 = (*(_WORD *)(v20 + 18) & 3) == 2;
    v71 = v19;
    v23 = sub_1389B50(&v71);
    v24 = v71 & 0xFFFFFFFFFFFFFFF8LL;
    v65 = v71 & 0xFFFFFFFFFFFFFFF8LL;
    if ( -1431655765
       * (unsigned int)((__int64)(v23
                                - ((v71 & 0xFFFFFFFFFFFFFFF8LL)
                                 - 24LL * (*(_DWORD *)((v71 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3) )
    {
      if ( *(_BYTE *)(v24 + 16) == 78 )
      {
        v26 = *(_QWORD *)(*(_QWORD *)(v24 + 40) + 8LL);
        v59 = *(_QWORD *)(v24 + 40);
        if ( v26 )
        {
          while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v26) + 16) - 25) > 9u )
          {
            v26 = *(_QWORD *)(v26 + 8);
            if ( !v26 )
              goto LABEL_24;
          }
          v62 = v12;
          v29 = v26;
          v72 = v74;
          v73 = 0x200000000LL;
          v30 = 0;
          while ( 1 )
          {
            v29 = *(_QWORD *)(v29 + 8);
            if ( !v29 )
              break;
            while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v29) + 16) - 25) <= 9u )
            {
              v29 = *(_QWORD *)(v29 + 8);
              ++v30;
              if ( !v29 )
                goto LABEL_35;
            }
          }
LABEL_35:
          v12 = v62;
          v63 = v30 + 1;
          if ( v30 + 1 > 2 )
          {
            sub_16CD150((__int64)&v72, v74, v63, 8, v27, v28);
            v31 = &v72[8 * (unsigned int)v73];
          }
          else
          {
            v31 = v74;
          }
          v32 = sub_1648700(v26);
LABEL_40:
          if ( v31 )
            *v31 = v32[5];
          while ( 1 )
          {
            v26 = *(_QWORD *)(v26 + 8);
            if ( !v26 )
              break;
            v32 = sub_1648700(v26);
            if ( (unsigned __int8)(*((_BYTE *)v32 + 16) - 25) <= 9u )
            {
              ++v31;
              goto LABEL_40;
            }
          }
          LODWORD(v73) = v73 + v63;
          v33 = (unsigned __int64)v72;
          if ( (_DWORD)v73 == 2
            && *(_BYTE *)(sub_157EBA0(*(_QWORD *)v72) + 16) != 28
            && *(_BYTE *)(sub_157EBA0(*(_QWORD *)(v33 + 8)) + 16) != 28 )
          {
            if ( sub_157F5F0(v59) )
            {
              v34 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v59) + 16) - 34;
              if ( (unsigned int)v34 > 0x36 || (v37 = 0x40018000000001LL, !_bittest64(&v37, v34)) )
              {
                v66 = v65 + 24;
                if ( v66 == *(_QWORD *)(v59 + 48) )
                {
LABEL_64:
                  if ( v72 != v74 )
                    _libc_free((unsigned __int64)v72);
                  v51 = sub_1A80E20(v71, a4, a5, a6, a7, a8, v35, v36, a11, a12);
                  if ( v51 )
                    v69 = v51;
                  else
                    v69 |= sub_1A81640(v71, a4, a5, a6, a7, a8, v52, v53, a11, a12);
                  goto LABEL_24;
                }
                v55 = v12;
                v38 = 0;
                v54 = v14;
                v39 = *(_QWORD *)(v59 + 48);
                while ( 1 )
                {
                  if ( !v39 )
                    BUG();
                  v40 = 24LL * (*(_DWORD *)(v39 - 4) & 0xFFFFFFF);
                  if ( (*(_BYTE *)(v39 - 1) & 0x40) != 0 )
                  {
                    v41 = *(char **)(v39 - 32);
                    v42 = &v41[v40];
                  }
                  else
                  {
                    v41 = (char *)(v39 - 24 - v40);
                    v42 = (char *)(v39 - 24);
                  }
                  v43 = v42 - v41;
                  v75 = v77;
                  v76 = 0x400000000LL;
                  v44 = 0xAAAAAAAAAAAAAAABLL * (v43 >> 3);
                  v45 = v44;
                  if ( (unsigned __int64)v43 > 0x60 )
                  {
                    v56 = v41;
                    v57 = v43;
                    v60 = 0xAAAAAAAAAAAAAAABLL * (v43 >> 3);
                    sub_16CD150((__int64)&v75, v77, v44, 8, v44, v43);
                    v48 = v75;
                    v47 = v76;
                    LODWORD(v44) = v60;
                    v45 = v60;
                    v43 = v57;
                    v41 = v56;
                    v46 = &v75[8 * (unsigned int)v76];
                  }
                  else
                  {
                    v46 = v77;
                    v47 = 0;
                    v48 = v77;
                  }
                  if ( v43 > 0 )
                  {
                    do
                    {
                      v49 = *(_QWORD *)v41;
                      v46 += 8;
                      v41 += 24;
                      *((_QWORD *)v46 - 1) = v49;
                      --v45;
                    }
                    while ( v45 );
                    v48 = v75;
                    v47 = v76;
                  }
                  LODWORD(v76) = v47 + v44;
                  v50 = sub_14A5330(a3, v39 - 24, (__int64)v48, (unsigned int)(v47 + v44));
                  if ( v75 != v77 )
                    _libc_free((unsigned __int64)v75);
                  v38 += v50;
                  if ( v38 >= dword_4FB4F00 )
                    break;
                  v39 = *(_QWORD *)(v39 + 8);
                  if ( v66 == v39 )
                  {
                    v12 = v55;
                    v14 = v54;
                    goto LABEL_64;
                  }
                }
                v12 = v55;
                v14 = v54;
              }
            }
            v33 = (unsigned __int64)v72;
          }
          if ( (_BYTE *)v33 != v74 )
            _libc_free(v33);
        }
      }
    }
LABEL_24:
    if ( v70 )
      goto LABEL_25;
    goto LABEL_5;
  }
  return v69;
}
