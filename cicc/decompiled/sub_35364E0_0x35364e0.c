// Function: sub_35364E0
// Address: 0x35364e0
//
void __fastcall sub_35364E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  void (__fastcall *v7)(__int64 **, __int64, __int64, unsigned int *); // rax
  __int64 v8; // rdx
  unsigned int v9; // esi
  unsigned int v10; // r13d
  __int64 v11; // r9
  __int64 v12; // r11
  int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 i; // rax
  bool v22; // zf
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // r8
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // eax
  _BYTE *v36; // rdi
  unsigned __int64 v37; // rcx
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  int v43; // ebx
  int v44; // ebx
  __int64 v45; // rax
  void *v46; // r13
  unsigned __int64 v47; // rdx
  size_t v48; // r12
  int v49; // eax
  __int64 v50; // rbx
  unsigned __int64 v51; // rcx
  __int64 v52; // rax
  void *v53; // r13
  int v54; // r8d
  __int64 v55; // r11
  __int64 v56; // rcx
  int v57; // edx
  int v58; // edi
  int v59; // edi
  int v60; // edi
  int v61; // ecx
  __int64 v62; // r10
  __int64 v63; // rsi
  int v64; // edi
  __int64 v65; // rsi
  __int64 *v66; // [rsp+18h] [rbp-128h]
  unsigned int v67; // [rsp+20h] [rbp-120h]
  __int64 v68; // [rsp+20h] [rbp-120h]
  unsigned int v69; // [rsp+20h] [rbp-120h]
  __int64 v70; // [rsp+20h] [rbp-120h]
  unsigned int v71; // [rsp+28h] [rbp-118h]
  __int64 *v72; // [rsp+28h] [rbp-118h]
  char v73; // [rsp+3Eh] [rbp-102h] BYREF
  char v74; // [rsp+3Fh] [rbp-101h] BYREF
  unsigned int v75; // [rsp+40h] [rbp-100h] BYREF
  int v76; // [rsp+44h] [rbp-FCh] BYREF
  __int64 v77; // [rsp+48h] [rbp-F8h] BYREF
  __int64 *v78; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v79; // [rsp+58h] [rbp-E8h]
  _QWORD v80[6]; // [rsp+60h] [rbp-E0h] BYREF
  void *v81; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v82; // [rsp+98h] [rbp-A8h]
  _BYTE v83[48]; // [rsp+A0h] [rbp-A0h] BYREF
  void *src; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v85; // [rsp+D8h] [rbp-68h]
  _BYTE v86[96]; // [rsp+E0h] [rbp-60h] BYREF

  v6 = *(_QWORD *)a3;
  v75 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, unsigned int *))(v6 + 1392))(a3, a2, &v75) )
  {
    v7 = *(void (__fastcall **)(__int64 **, __int64, __int64, unsigned int *))(*(_QWORD *)a3 + 1400LL);
    if ( (char *)v7 == (char *)sub_2FDCE70 )
    {
      v8 = *(_QWORD *)(a2 + 56);
      v80[1] = a2 + 48;
      v78 = v80;
      v80[0] = v8;
      v79 = 0x300000001LL;
    }
    else
    {
      v7(&v78, a3, a2, &v75);
      if ( !(_DWORD)v79 )
      {
LABEL_36:
        if ( v78 != v80 )
          _libc_free((unsigned __int64)v78);
        return;
      }
    }
    v9 = *(_DWORD *)(a1 + 72);
    v10 = v75;
    v11 = a1 + 48;
    if ( v9 )
    {
      v12 = *(_QWORD *)(a1 + 56);
      v13 = 1;
      v71 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
      v14 = 0;
      v15 = (v9 - 1) & v71;
      v16 = v12 + 16 * v15;
      v17 = *(_QWORD *)v16;
      if ( a2 == *(_QWORD *)v16 )
        goto LABEL_6;
      while ( v17 != -4096 )
      {
        if ( !v14 && v17 == -8192 )
          v14 = v16;
        v15 = (v9 - 1) & (v13 + (_DWORD)v15);
        v16 = v12 + 16LL * (unsigned int)v15;
        v17 = *(_QWORD *)v16;
        if ( a2 == *(_QWORD *)v16 )
          goto LABEL_6;
        ++v13;
      }
      v58 = *(_DWORD *)(a1 + 64);
      if ( v14 )
        v16 = v14;
      ++*(_QWORD *)(a1 + 48);
      v57 = v58 + 1;
      if ( 4 * (v58 + 1) < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 68) - v57 > v9 >> 3 )
        {
LABEL_75:
          *(_DWORD *)(a1 + 64) = v57;
          if ( *(_QWORD *)v16 != -4096 )
            --*(_DWORD *)(a1 + 68);
          *(_QWORD *)v16 = a2;
          *(_DWORD *)(v16 + 8) = 0;
LABEL_6:
          *(_DWORD *)(v16 + 8) = v10;
          v18 = *(_QWORD *)(a2 + 56);
          v73 = 0;
          v77 = v18;
          v81 = v83;
          v82 = 0xC00000000LL;
          src = v86;
          v85 = 0x600000000LL;
          v76 = 0;
          v74 = 0;
          v66 = &v78[2 * (unsigned int)v79];
          if ( v66 == v78 )
            goto LABEL_49;
          v72 = v78;
          while ( 1 )
          {
            v19 = *v72;
            v20 = v72[1];
            for ( i = v77; v19 != i; v77 = i )
            {
              while ( 1 )
              {
                v22 = *(_BYTE *)(a1 + 208) == 0;
                v74 = 0;
                if ( v22 )
                {
                  v23 = (unsigned int)v85;
                  v24 = HIDWORD(v85);
                  *(_BYTE *)(a1 + 208) = 1;
                  if ( v23 + 1 > v24 )
                  {
                    v68 = i;
                    sub_C8D5F0((__int64)&src, v86, v23 + 1, 8u, v23 + 1, v11);
                    v23 = (unsigned int)v85;
                    i = v68;
                  }
                  *((_QWORD *)src + v23) = i;
                  v25 = (unsigned int)v82;
                  LODWORD(v85) = v85 + 1;
                  v26 = (unsigned int)v82 + 1LL;
                  v15 = *(unsigned int *)(a1 + 8);
                  if ( v26 > HIDWORD(v82) )
                  {
                    v67 = *(_DWORD *)(a1 + 8);
                    sub_C8D5F0((__int64)&v81, v83, v26, 4u, v15, v11);
                    v25 = (unsigned int)v82;
                    v15 = v67;
                  }
                  *((_DWORD *)v81 + v25) = v15;
                  LODWORD(v82) = v82 + 1;
                  --*(_DWORD *)(a1 + 8);
                }
                v27 = v77;
                if ( !v77 )
                  BUG();
                if ( (*(_BYTE *)v77 & 4) == 0 )
                  break;
                i = *(_QWORD *)(v77 + 8);
                v77 = i;
                if ( v19 == i )
                  goto LABEL_20;
              }
              while ( (*(_BYTE *)(v27 + 44) & 8) != 0 )
                v27 = *(_QWORD *)(v27 + 8);
              i = *(_QWORD *)(v27 + 8);
            }
LABEL_20:
            if ( v20 != i )
              break;
LABEL_45:
            v72 += 2;
            if ( v66 == v72 )
            {
              if ( v73 )
              {
                v22 = *(_BYTE *)(a1 + 208) == 0;
                v74 = 0;
                if ( v22 )
                {
                  v38 = (unsigned int)v85;
                  v39 = HIDWORD(v85);
                  *(_BYTE *)(a1 + 208) = 1;
                  v40 = v77;
                  if ( v38 + 1 > v39 )
                  {
                    sub_C8D5F0((__int64)&src, v86, v38 + 1, 8u, v15, v11);
                    v38 = (unsigned int)v85;
                  }
                  *((_QWORD *)src + v38) = v40;
                  v41 = (unsigned int)v82;
                  LODWORD(v85) = v85 + 1;
                  v42 = (unsigned int)v82 + 1LL;
                  v43 = *(_DWORD *)(a1 + 8);
                  if ( v42 > HIDWORD(v82) )
                  {
                    sub_C8D5F0((__int64)&v81, v83, v42, 4u, v15, v11);
                    v41 = (unsigned int)v82;
                  }
                  *((_DWORD *)v81 + v41) = v43;
                  LODWORD(v82) = v82 + 1;
                  --*(_DWORD *)(a1 + 8);
                }
                v44 = v85;
                v45 = *(unsigned int *)(a1 + 152);
                v46 = src;
                v47 = v45 + (unsigned int)v85;
                v48 = 8LL * (unsigned int)v85;
                if ( v47 > *(unsigned int *)(a1 + 156) )
                {
                  sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), v47, 8u, v15, v11);
                  v45 = *(unsigned int *)(a1 + 152);
                }
                if ( v48 )
                {
                  memcpy((void *)(*(_QWORD *)(a1 + 144) + 8 * v45), v46, v48);
                  LODWORD(v45) = *(_DWORD *)(a1 + 152);
                }
                v49 = v44 + v45;
                v50 = (unsigned int)v82;
                v51 = *(unsigned int *)(a1 + 92);
                *(_DWORD *)(a1 + 152) = v49;
                v52 = *(unsigned int *)(a1 + 88);
                v53 = v81;
                if ( v52 + v50 > v51 )
                {
                  sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v52 + v50, 4u, v15, v11);
                  v52 = *(unsigned int *)(a1 + 88);
                }
                if ( 4 * v50 )
                {
                  memcpy((void *)(*(_QWORD *)(a1 + 80) + 4 * v52), v53, 4 * v50);
                  LODWORD(v52) = *(_DWORD *)(a1 + 88);
                }
                v36 = src;
                *(_DWORD *)(a1 + 88) = v50 + v52;
                if ( v36 == v86 )
                  goto LABEL_49;
              }
              else
              {
                v36 = src;
                if ( src == v86 )
                  goto LABEL_49;
              }
              _libc_free((unsigned __int64)v36);
LABEL_49:
              if ( v81 != v83 )
                _libc_free((unsigned __int64)v81);
              goto LABEL_36;
            }
          }
          while ( 1 )
          {
            v35 = sub_2FE0CE0(a3, *(_QWORD *)a1, &v77, v75);
            if ( v35 == 2 )
            {
              v22 = *(_BYTE *)(a1 + 208) == 0;
              v74 = 0;
              if ( v22 )
              {
                v28 = (unsigned int)v85;
                v37 = HIDWORD(v85);
                *(_BYTE *)(a1 + 208) = 1;
                v30 = v77;
                v31 = v28 + 1;
                if ( v28 + 1 > v37 )
                {
LABEL_24:
                  v70 = v77;
                  sub_C8D5F0((__int64)&src, v86, v31, 8u, v77, v11);
                  v28 = (unsigned int)v85;
                  v30 = v70;
                }
LABEL_25:
                *((_QWORD *)src + v28) = v30;
                v32 = (unsigned int)v82;
                LODWORD(v85) = v85 + 1;
                v33 = (unsigned int)v82 + 1LL;
                v15 = *(unsigned int *)(a1 + 8);
                if ( v33 > HIDWORD(v82) )
                {
                  v69 = *(_DWORD *)(a1 + 8);
                  sub_C8D5F0((__int64)&v81, v83, v33, 4u, v15, v11);
                  v32 = (unsigned int)v82;
                  v15 = v69;
                }
                *((_DWORD *)v81 + v32) = v15;
                LODWORD(v82) = v82 + 1;
                --*(_DWORD *)(a1 + 8);
              }
            }
            else
            {
              if ( v35 > 2 )
              {
                if ( v35 == 3 )
                  *(_BYTE *)(a1 + 208) = 0;
                goto LABEL_28;
              }
              if ( v35 )
              {
                sub_35360E0(a1, &v77, &v74, &v73, &v76, (__int64)&v81, (__int64)&src);
                v22 = *(_BYTE *)(a1 + 208) == 0;
                v74 = 0;
                if ( v22 )
                {
                  v28 = (unsigned int)v85;
                  v29 = HIDWORD(v85);
                  *(_BYTE *)(a1 + 208) = 1;
                  v30 = v77;
                  v31 = v28 + 1;
                  if ( v28 + 1 > v29 )
                    goto LABEL_24;
                  goto LABEL_25;
                }
              }
              else
              {
                sub_35360E0(a1, &v77, &v74, &v73, &v76, (__int64)&v81, (__int64)&src);
              }
            }
LABEL_28:
            v34 = v77;
            if ( !v77 )
              BUG();
            if ( (*(_BYTE *)v77 & 4) != 0 )
            {
              v77 = *(_QWORD *)(v77 + 8);
              if ( v20 == v77 )
                goto LABEL_45;
            }
            else
            {
              while ( (*(_BYTE *)(v34 + 44) & 8) != 0 )
                v34 = *(_QWORD *)(v34 + 8);
              v77 = *(_QWORD *)(v34 + 8);
              if ( v20 == v77 )
                goto LABEL_45;
            }
          }
        }
        sub_2EB73C0(a1 + 48, v9);
        v59 = *(_DWORD *)(a1 + 72);
        if ( v59 )
        {
          v60 = v59 - 1;
          v15 = *(_QWORD *)(a1 + 56);
          v11 = 0;
          v57 = *(_DWORD *)(a1 + 64) + 1;
          v61 = 1;
          LODWORD(v62) = v60 & v71;
          v16 = v15 + 16LL * (v60 & v71);
          v63 = *(_QWORD *)v16;
          if ( a2 != *(_QWORD *)v16 )
          {
            while ( v63 != -4096 )
            {
              if ( v63 == -8192 && !v11 )
                v11 = v16;
              v62 = v60 & (unsigned int)(v62 + v61);
              v16 = v15 + 16 * v62;
              v63 = *(_QWORD *)v16;
              if ( a2 == *(_QWORD *)v16 )
                goto LABEL_75;
              ++v61;
            }
            if ( v11 )
              v16 = v11;
          }
          goto LABEL_75;
        }
LABEL_111:
        ++*(_DWORD *)(a1 + 64);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 48);
    }
    sub_2EB73C0(a1 + 48, 2 * v9);
    v54 = *(_DWORD *)(a1 + 72);
    if ( v54 )
    {
      v15 = (unsigned int)(v54 - 1);
      v55 = *(_QWORD *)(a1 + 56);
      LODWORD(v56) = v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v57 = *(_DWORD *)(a1 + 64) + 1;
      v16 = v55 + 16LL * (unsigned int)v56;
      v11 = *(_QWORD *)v16;
      if ( a2 != *(_QWORD *)v16 )
      {
        v64 = 1;
        v65 = 0;
        while ( v11 != -4096 )
        {
          if ( !v65 && v11 == -8192 )
            v65 = v16;
          v56 = (unsigned int)v15 & ((_DWORD)v56 + v64);
          v16 = v55 + 16 * v56;
          v11 = *(_QWORD *)v16;
          if ( a2 == *(_QWORD *)v16 )
            goto LABEL_75;
          ++v64;
        }
        if ( v65 )
          v16 = v65;
      }
      goto LABEL_75;
    }
    goto LABEL_111;
  }
}
