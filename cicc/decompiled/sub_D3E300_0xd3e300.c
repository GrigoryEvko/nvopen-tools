// Function: sub_D3E300
// Address: 0xd3e300
//
__int64 __fastcall sub_D3E300(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 *a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        _DWORD *a8,
        int a9,
        char a10)
{
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 *v14; // rdi
  __int64 v15; // r9
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 *v20; // rbx
  __int64 v21; // r11
  __int64 v23; // rax
  __int64 *v24; // r10
  size_t v25; // rdx
  __int64 *v26; // r13
  __int64 v27; // r14
  __int64 v28; // r12
  int v29; // eax
  unsigned __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdi
  unsigned __int64 *v33; // rbx
  unsigned __int64 *v34; // rdi
  unsigned __int64 *v35; // rdx
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rsi
  _QWORD *v40; // r9
  unsigned __int64 v41; // r11
  unsigned int v42; // esi
  __int64 v43; // r9
  __int64 *v44; // rax
  unsigned __int64 v45; // rdx
  int *v46; // rdx
  unsigned __int64 *v47; // r10
  int v48; // edi
  _QWORD *v49; // r11
  _QWORD *v50; // r10
  _QWORD *v51; // rax
  _BYTE *v52; // rdi
  _BYTE *v53; // rax
  int v54; // eax
  __int64 v55; // rdx
  int v56; // ebx
  unsigned int v57; // esi
  unsigned __int64 v58; // rax
  unsigned __int64 *v59; // r9
  int v60; // eax
  __int64 v61; // r9
  unsigned int v62; // esi
  unsigned __int64 v63; // rdx
  int v64; // eax
  unsigned __int64 *v65; // [rsp+8h] [rbp-128h]
  _QWORD *v66; // [rsp+10h] [rbp-120h]
  _QWORD *v67; // [rsp+18h] [rbp-118h]
  _QWORD *v68; // [rsp+20h] [rbp-110h]
  int v69; // [rsp+20h] [rbp-110h]
  unsigned __int64 v70; // [rsp+28h] [rbp-108h]
  int v71; // [rsp+30h] [rbp-100h]
  unsigned __int64 v72; // [rsp+30h] [rbp-100h]
  unsigned __int64 v73; // [rsp+30h] [rbp-100h]
  unsigned __int64 v74; // [rsp+30h] [rbp-100h]
  unsigned int v76; // [rsp+38h] [rbp-F8h]
  char v77; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v78; // [rsp+38h] [rbp-F8h]
  char v79; // [rsp+38h] [rbp-F8h]
  unsigned int v83; // [rsp+58h] [rbp-D8h]
  _QWORD *v84; // [rsp+58h] [rbp-D8h]
  char v85; // [rsp+58h] [rbp-D8h]
  int v86; // [rsp+58h] [rbp-D8h]
  unsigned __int8 v87; // [rsp+6Ch] [rbp-C4h]
  char v88; // [rsp+6Ch] [rbp-C4h]
  unsigned __int64 v89; // [rsp+70h] [rbp-C0h]
  __int64 v90; // [rsp+78h] [rbp-B8h]
  __int64 *v91; // [rsp+78h] [rbp-B8h]
  __int64 *v92; // [rsp+78h] [rbp-B8h]
  __int64 *v93; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v94; // [rsp+88h] [rbp-A8h]
  _QWORD v95[6]; // [rsp+90h] [rbp-A0h] BYREF
  void *src; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v97; // [rsp+C8h] [rbp-68h]
  _BYTE v98[96]; // [rsp+D0h] [rbp-60h] BYREF

  v12 = *(_QWORD *)(a1 + 1072);
  v87 = a10;
  v13 = *(_QWORD *)(v12 + 112);
  v89 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  src = v98;
  v97 = 0x600000000LL;
  v90 = v13;
  sub_D37810(v13, a7, a3 & 0xFFFFFFFFFFFFFFF8LL, (__int64)&src, qword_4F86F88);
  if ( (_DWORD)v97 != 2 )
    goto LABEL_5;
  v14 = (__int64 *)src;
  v15 = v90;
  v16 = *(_QWORD *)src & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_WORD *)(v16 + 24) == 8 )
  {
    v17 = (_QWORD *)(*((_QWORD *)src + 1) & 0xFFFFFFFFFFFFFFF8LL);
    if ( *((_WORD *)v17 + 12) == 8 )
      goto LABEL_29;
  }
  else
  {
    if ( !(unsigned __int8)sub_DADE90(v90, v16, a7) )
      goto LABEL_5;
    v14 = (__int64 *)src;
    v15 = v90;
    v17 = (_QWORD *)(*((_QWORD *)src + 1) & 0xFFFFFFFFFFFFFFF8LL);
    if ( *((_WORD *)v17 + 12) == 8 )
      goto LABEL_29;
  }
  if ( !(unsigned __int8)sub_DADE90(v15, v17, a7) )
  {
LABEL_5:
    v17 = (_QWORD *)a5;
    v18 = sub_D34370(v12, a5, v89);
    v14 = (__int64 *)src;
    v95[0] = v18 & 0xFFFFFFFFFFFFFFFBLL;
    v93 = v95;
    v94 = 0x600000001LL;
LABEL_6:
    if ( v14 != (__int64 *)v98 )
      _libc_free(v14, v17);
    v14 = v93;
    v19 = (unsigned int)v94;
    goto LABEL_9;
  }
  v14 = (__int64 *)src;
LABEL_29:
  v19 = (unsigned int)v97;
  v93 = v95;
  v94 = 0x600000000LL;
  if ( !(_DWORD)v97 )
    goto LABEL_6;
  if ( v14 == (__int64 *)v98 )
  {
    v24 = v95;
    v25 = 8LL * (unsigned int)v97;
    if ( (unsigned int)v97 <= 6
      || (v17 = v95,
          sub_C8D5F0((__int64)&v93, v95, (unsigned int)v97, 8u, (__int64)&v93, (unsigned int)v97),
          v24 = v93,
          v14 = (__int64 *)src,
          (v25 = 8LL * (unsigned int)v97) != 0) )
    {
      v17 = v14;
      memcpy(v24, v14, v25);
      v14 = (__int64 *)src;
    }
    LODWORD(v94) = v19;
    goto LABEL_6;
  }
  v93 = v14;
  v94 = v97;
LABEL_9:
  v91 = &v14[v19];
  if ( v91 != v14 )
  {
    v20 = v14;
    do
    {
      v17 = (_QWORD *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
      if ( !(unsigned __int8)sub_DADE90(*(_QWORD *)(*(_QWORD *)(a1 + 1072) + 112LL), v17, a7) )
      {
        v17 = (_QWORD *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
        if ( *((_WORD *)v17 + 12) != 8 )
        {
          if ( !a10 )
          {
            v14 = v93;
            goto LABEL_19;
          }
          v17 = (_QWORD *)sub_DEF530(*(_QWORD *)(a1 + 1072), v89);
          if ( !v17 )
            goto LABEL_23;
        }
        if ( v17[5] != 2 )
          goto LABEL_23;
        v21 = 0;
        if ( (_DWORD)v94 == 1 )
        {
          v23 = sub_D34370(*(_QWORD *)(a1 + 1072), a5, v89);
          v21 = 0;
          v17 = (_QWORD *)v23;
          *v20 = v23 | *v20 & 7;
          if ( (_DWORD)v94 == 1 )
            v21 = v89;
        }
        if ( !sub_D34050(*(_QWORD *)(a1 + 1072), (__int64)v17, v21, a4, a7, a10, (__int64)src, 0) )
        {
LABEL_23:
          v87 = 0;
          v14 = v93;
          goto LABEL_19;
        }
      }
      ++v20;
    }
    while ( v91 != v20 );
    v14 = v93;
    v92 = &v93[(unsigned int)v94];
    if ( v92 != v93 )
    {
      v26 = v93;
      v27 = a1;
      v28 = a3;
      v88 = (a3 >> 2) & 1;
      while ( 1 )
      {
        v30 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
        v31 = (*v26 >> 2) & 1;
        if ( *(_DWORD *)(v27 + 64) )
        {
          v32 = *(_QWORD *)(v27 + 1056);
          v33 = *(unsigned __int64 **)(v32 + 16);
          v34 = (unsigned __int64 *)(v32 + 8);
          if ( !v33 )
            goto LABEL_111;
          v35 = v34;
          do
          {
            while ( 1 )
            {
              v36 = v33[2];
              v37 = v33[3];
              if ( v28 <= (__int64)v33[6] )
                break;
              v33 = (unsigned __int64 *)v33[3];
              if ( !v37 )
                goto LABEL_44;
            }
            v35 = v33;
            v33 = (unsigned __int64 *)v33[2];
          }
          while ( v36 );
LABEL_44:
          if ( v34 == v35 || (v38 = v35[6], v28 < v38) )
LABEL_111:
            BUG();
          if ( (v35[5] & 1) == 0 )
          {
            v39 = v35[4];
            if ( (*(_BYTE *)(v39 + 8) & 1) != 0 )
            {
              v38 = *(_QWORD *)(v39 + 16);
            }
            else
            {
              v40 = *(_QWORD **)v39;
              if ( (*(_BYTE *)(*(_QWORD *)v39 + 8LL) & 1) == 0 )
              {
                v49 = (_QWORD *)*v40;
                if ( (*(_BYTE *)(*v40 + 8LL) & 1) != 0 )
                {
                  v40 = (_QWORD *)*v40;
                }
                else
                {
                  v50 = (_QWORD *)*v49;
                  if ( (*(_BYTE *)(*v49 + 8LL) & 1) == 0 )
                  {
                    v51 = (_QWORD *)*v50;
                    v84 = (_QWORD *)*v50;
                    if ( (*(_BYTE *)(*v50 + 8LL) & 1) != 0 )
                    {
                      v50 = (_QWORD *)*v50;
                    }
                    else
                    {
                      v52 = (_BYTE *)*v51;
                      if ( (*(_BYTE *)(*v51 + 8LL) & 1) == 0 )
                      {
                        v65 = v35;
                        v66 = (_QWORD *)*v49;
                        v67 = (_QWORD *)*v40;
                        v68 = *(_QWORD **)v39;
                        v72 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
                        v77 = (*v26 >> 2) & 1;
                        v53 = sub_D38E40(v52);
                        v35 = v65;
                        v50 = v66;
                        v52 = v53;
                        v49 = v67;
                        v40 = v68;
                        *v84 = v53;
                        v30 = v72;
                        LOBYTE(v31) = v77;
                      }
                      *v50 = v52;
                      v50 = v52;
                    }
                    *v49 = v50;
                  }
                  *v40 = v50;
                  v40 = v50;
                }
                *(_QWORD *)v39 = v40;
              }
              v35[4] = (unsigned __int64)v40;
              v38 = v40[2];
            }
          }
          v41 = v38 & 0xFFFFFFFFFFFFFFF8LL;
          v42 = *(_DWORD *)(a6 + 24);
          v43 = *(_QWORD *)(a6 + 8);
          if ( !v42 )
          {
            ++*(_QWORD *)a6;
            goto LABEL_77;
          }
          v83 = ((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4);
          v76 = (v42 - 1) & v83;
          v44 = (__int64 *)(v43 + 16LL * v76);
          v45 = *v44;
          if ( v41 != *v44 )
          {
            v71 = 1;
            v47 = 0;
            while ( v45 != -4096 )
            {
              if ( !v47 && v45 == -8192 )
                v47 = (unsigned __int64 *)v44;
              v76 = (v42 - 1) & (v76 + v71);
              v44 = (__int64 *)(v43 + 16LL * v76);
              v45 = *v44;
              if ( v41 == *v44 )
                goto LABEL_52;
              ++v71;
            }
            if ( !v47 )
              v47 = (unsigned __int64 *)v44;
            ++*(_QWORD *)a6;
            v48 = *(_DWORD *)(a6 + 16) + 1;
            if ( 4 * v48 >= 3 * v42 )
            {
LABEL_77:
              v78 = v30;
              v73 = v41;
              v85 = v31;
              sub_D39D40(a6, 2 * v42);
              v54 = *(_DWORD *)(a6 + 24);
              if ( !v54 )
                goto LABEL_112;
              v41 = v73;
              v55 = *(_QWORD *)(a6 + 8);
              v56 = v54 - 1;
              LOBYTE(v31) = v85;
              v48 = *(_DWORD *)(a6 + 16) + 1;
              v30 = v78;
              v57 = (v54 - 1) & (((unsigned int)v73 >> 4) ^ ((unsigned int)v73 >> 9));
              v47 = (unsigned __int64 *)(v55 + 16LL * v57);
              v58 = *v47;
              if ( v73 != *v47 )
              {
                v86 = 1;
                v59 = (unsigned __int64 *)(v55 + 16LL * v57);
                v47 = 0;
                while ( v58 != -4096 )
                {
                  if ( !v47 && v58 == -8192 )
                    v47 = v59;
                  v57 = v56 & (v86 + v57);
                  v59 = (unsigned __int64 *)(v55 + 16LL * v57);
                  v58 = *v59;
                  if ( v73 == *v59 )
                  {
                    v47 = (unsigned __int64 *)(v55 + 16LL * v57);
                    goto LABEL_63;
                  }
                  ++v86;
                }
                if ( !v47 )
                  v47 = v59;
              }
            }
            else if ( v42 - *(_DWORD *)(a6 + 20) - v48 <= v42 >> 3 )
            {
              v74 = v30;
              v70 = v41;
              v79 = v31;
              sub_D39D40(a6, v42);
              v60 = *(_DWORD *)(a6 + 24);
              if ( !v60 )
              {
LABEL_112:
                ++*(_DWORD *)(a6 + 16);
                BUG();
              }
              v61 = *(_QWORD *)(a6 + 8);
              v69 = v60 - 1;
              v41 = v70;
              v62 = (v60 - 1) & v83;
              LOBYTE(v31) = v79;
              v47 = (unsigned __int64 *)(v61 + 16LL * v62);
              v30 = v74;
              v63 = *v47;
              v48 = *(_DWORD *)(a6 + 16) + 1;
              v64 = 1;
              if ( v70 != *v47 )
              {
                while ( v63 != -4096 )
                {
                  if ( v63 == -8192 && !v33 )
                    v33 = v47;
                  v62 = v69 & (v64 + v62);
                  v47 = (unsigned __int64 *)(v61 + 16LL * v62);
                  v63 = *v47;
                  if ( v70 == *v47 )
                    goto LABEL_63;
                  ++v64;
                }
                if ( v33 )
                  v47 = v33;
              }
            }
LABEL_63:
            *(_DWORD *)(a6 + 16) = v48;
            if ( *v47 != -4096 )
              --*(_DWORD *)(a6 + 20);
            *v47 = v41;
            v46 = (int *)(v47 + 1);
            *((_DWORD *)v47 + 2) = 0;
LABEL_53:
            v29 = (*a8)++;
            *v46 = v29;
            goto LABEL_37;
          }
LABEL_52:
          v46 = (int *)(v44 + 1);
          v29 = *((_DWORD *)v44 + 2);
          if ( !v29 )
            goto LABEL_53;
        }
        else
        {
          v29 = (*a8)++;
        }
LABEL_37:
        v17 = (_QWORD *)a7;
        ++v26;
        sub_D3E0C0(a2, a7, v89, v30, (__int64)a4, v88, v29, a9, *(_QWORD *)(v27 + 1072), v31);
        if ( v92 == v26 )
        {
          v87 = 1;
          v14 = v93;
          goto LABEL_19;
        }
      }
    }
  }
  v87 = 1;
LABEL_19:
  if ( v14 != v95 )
    _libc_free(v14, v17);
  return v87;
}
