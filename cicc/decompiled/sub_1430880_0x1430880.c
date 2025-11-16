// Function: sub_1430880
// Address: 0x1430880
//
__int64 __fastcall sub_1430880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rdi
  unsigned int v5; // eax
  char v6; // dl
  __int64 *v7; // r13
  __int64 **v8; // rax
  __int64 **v9; // rcx
  unsigned int v10; // edi
  __int64 **v11; // rsi
  unsigned __int8 v12; // al
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rcx
  __int64 *v16; // r14
  unsigned __int64 v17; // rax
  __int64 v18; // r12
  unsigned __int8 v19; // al
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rbx
  _QWORD *v22; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r15
  unsigned __int64 v25; // r15
  unsigned int v26; // esi
  unsigned __int64 v27; // r15
  __int64 v28; // rdi
  _QWORD *v29; // r9
  int v30; // r11d
  unsigned int v31; // ecx
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  int v36; // eax
  char *v37; // rsi
  int v38; // ebx
  int v39; // ebx
  __int64 v40; // r10
  unsigned __int32 v41; // r8d
  unsigned __int64 v42; // rdx
  int v43; // esi
  _QWORD *v44; // rcx
  int v45; // ebx
  int v46; // ebx
  __int64 v47; // r10
  unsigned __int32 v48; // r8d
  int v49; // esi
  unsigned __int64 v50; // rdx
  unsigned __int8 v51; // [rsp+Fh] [rbp-251h]
  unsigned __int64 v52; // [rsp+10h] [rbp-250h]
  unsigned __int64 v53; // [rsp+18h] [rbp-248h]
  unsigned __int64 v54; // [rsp+20h] [rbp-240h]
  unsigned __int64 v55; // [rsp+28h] [rbp-238h]
  _QWORD *v56; // [rsp+28h] [rbp-238h]
  unsigned __int64 v60; // [rsp+50h] [rbp-210h] BYREF
  unsigned __int64 v61[2]; // [rsp+60h] [rbp-200h] BYREF
  __int64 v62; // [rsp+70h] [rbp-1F0h] BYREF
  __m128i v63; // [rsp+80h] [rbp-1E0h] BYREF
  _QWORD *v64; // [rsp+90h] [rbp-1D0h]
  _QWORD *v65; // [rsp+98h] [rbp-1C8h]
  __int64 v66; // [rsp+A0h] [rbp-1C0h]
  _QWORD *v67; // [rsp+120h] [rbp-140h] BYREF
  __int64 v68; // [rsp+128h] [rbp-138h]
  _QWORD v69[38]; // [rsp+130h] [rbp-130h] BYREF

  v4 = v69;
  v68 = 0x2000000001LL;
  v5 = 1;
  v67 = v69;
  v69[0] = a2;
  v51 = 0;
  while ( v5 )
  {
    v7 = (__int64 *)v4[v5 - 1];
    LODWORD(v68) = v5 - 1;
    v8 = *(__int64 ***)(a4 + 8);
    if ( *(__int64 ***)(a4 + 16) != v8 )
      goto LABEL_2;
    v9 = &v8[*(unsigned int *)(a4 + 28)];
    v10 = *(_DWORD *)(a4 + 28);
    if ( v8 == v9 )
      goto LABEL_53;
    v11 = 0;
    do
    {
      while ( 1 )
      {
        if ( v7 == *v8 )
          goto LABEL_3;
        if ( *v8 != (__int64 *)-2LL )
          break;
        v11 = v8;
        if ( v8 + 1 == v9 )
          goto LABEL_12;
        ++v8;
      }
      ++v8;
    }
    while ( v9 != v8 );
    if ( !v11 )
    {
LABEL_53:
      if ( v10 >= *(_DWORD *)(a4 + 24) )
      {
LABEL_2:
        sub_16CCBA0(a4, v7);
        if ( !v6 )
          goto LABEL_3;
        goto LABEL_13;
      }
      *(_DWORD *)(a4 + 28) = v10 + 1;
      *v9 = v7;
      ++*(_QWORD *)a4;
    }
    else
    {
LABEL_12:
      *v11 = v7;
      --*(_DWORD *)(a4 + 32);
      ++*(_QWORD *)a4;
    }
LABEL_13:
    v12 = *((_BYTE *)v7 + 16);
    v13 = 0;
    if ( v12 > 0x17u )
    {
      if ( v12 == 78 )
      {
        v13 = (unsigned __int64)v7 | 4;
      }
      else
      {
        v13 = (unsigned __int64)v7 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v12 != 29 )
          v13 = 0;
      }
    }
    v14 = 24LL * (*((_DWORD *)v7 + 5) & 0xFFFFFFF);
    v15 = &v7[v14 / 0xFFFFFFFFFFFFFFF8LL];
    if ( (*((_BYTE *)v7 + 23) & 0x40) != 0 )
    {
      v15 = (__int64 *)*(v7 - 1);
      v7 = &v15[(unsigned __int64)v14 / 8];
    }
    if ( v15 != v7 )
    {
      v16 = v15;
      v53 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v17 = (v13 & 0xFFFFFFFFFFFFFFF8LL) - 72;
      if ( (v13 & 4) != 0 )
        v17 = (v13 & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v52 = v17;
      do
      {
        v18 = *v16;
        v19 = *(_BYTE *)(*v16 + 16);
        if ( (unsigned __int8)(v19 - 17) > 6u )
        {
          if ( v19 == 4 )
          {
            v51 = 1;
          }
          else
          {
            if ( v19 > 3u )
            {
              v34 = (unsigned int)v68;
              if ( (unsigned int)v68 >= HIDWORD(v68) )
              {
                sub_16CD150(&v67, v69, 0, 8);
                v34 = (unsigned int)v68;
              }
              v67[v34] = v18;
              LODWORD(v68) = v68 + 1;
              goto LABEL_41;
            }
            if ( !v53 || (__int64 *)v52 != v16 )
            {
              sub_15E4EB0(v61);
              v20 = v61[0];
              v55 = v61[1];
              sub_16C1840(&v63);
              sub_16C1A90(&v63, v20, v55);
              sub_16C1AA0(&v63, &v60);
              v21 = v60;
              if ( (__int64 *)v61[0] != &v62 )
                j_j___libc_free_0(v61[0], v62 + 1);
              v61[0] = v21;
              if ( *(_BYTE *)(a1 + 178) )
              {
                v63.m128i_i64[0] = 0;
              }
              else
              {
                v63.m128i_i64[1] = 0;
                v63.m128i_i64[0] = (__int64)byte_3F871B3;
              }
              v64 = 0;
              v65 = 0;
              v66 = 0;
              v22 = sub_142DA40((_QWORD *)a1, v61, &v63);
              v23 = v65;
              v24 = v64;
              v56 = v22;
              v54 = (unsigned __int64)(v22 + 4);
              if ( v65 != v64 )
              {
                do
                {
                  if ( *v24 )
                    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v24 + 8LL))(*v24);
                  ++v24;
                }
                while ( v23 != v24 );
                v24 = v64;
              }
              if ( v24 )
                j_j___libc_free_0(v24, v66 - (_QWORD)v24);
              v56[5] = v18;
              v25 = (4LL * *(unsigned __int8 *)(a1 + 178)) | v54 & 0xFFFFFFFFFFFFFFFBLL;
              v63.m128i_i64[0] = v25;
              v26 = *(_DWORD *)(a3 + 24);
              if ( !v26 )
              {
                ++*(_QWORD *)a3;
                goto LABEL_73;
              }
              v27 = v25 & 0xFFFFFFFFFFFFFFF8LL;
              v28 = *(_QWORD *)(a3 + 8);
              v29 = 0;
              v30 = 1;
              v31 = v27 & (v26 - 1);
              v32 = (_QWORD *)(v28 + 8LL * v31);
              v33 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v27 != v33 )
              {
                while ( v33 != -8 )
                {
                  if ( v33 != -16 || v29 )
                    v32 = v29;
                  v31 = (v26 - 1) & (v30 + v31);
                  v33 = *(_QWORD *)(v28 + 8LL * v31) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v27 == v33 )
                    goto LABEL_41;
                  ++v30;
                  v29 = v32;
                  v32 = (_QWORD *)(v28 + 8LL * v31);
                }
                if ( !v29 )
                  v29 = v32;
                ++*(_QWORD *)a3;
                v36 = *(_DWORD *)(a3 + 16) + 1;
                if ( 4 * v36 >= 3 * v26 )
                {
LABEL_73:
                  sub_14306B0(a3, 2 * v26);
                  v38 = *(_DWORD *)(a3 + 24);
                  if ( !v38 )
                    goto LABEL_94;
                  v39 = v38 - 1;
                  v40 = *(_QWORD *)(a3 + 8);
                  v41 = v63.m128i_i32[0] & 0xFFFFFFF8 & v39;
                  v29 = (_QWORD *)(v40 + 8LL * v41);
                  v36 = *(_DWORD *)(a3 + 16) + 1;
                  v42 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (v63.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != v42 )
                  {
                    v43 = 1;
                    v44 = 0;
                    while ( v42 != -8 )
                    {
                      if ( v42 == -16 && !v44 )
                        v44 = v29;
                      v41 = v39 & (v43 + v41);
                      v29 = (_QWORD *)(v40 + 8LL * v41);
                      v42 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (v63.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == v42 )
                        goto LABEL_65;
                      ++v43;
                    }
LABEL_77:
                    if ( v44 )
                      v29 = v44;
                  }
                }
                else if ( v26 - *(_DWORD *)(a3 + 20) - v36 <= v26 >> 3 )
                {
                  sub_14306B0(a3, v26);
                  v45 = *(_DWORD *)(a3 + 24);
                  if ( !v45 )
                  {
LABEL_94:
                    ++*(_DWORD *)(a3 + 16);
                    BUG();
                  }
                  v46 = v45 - 1;
                  v47 = *(_QWORD *)(a3 + 8);
                  v44 = 0;
                  v48 = v63.m128i_i32[0] & 0xFFFFFFF8 & v46;
                  v29 = (_QWORD *)(v47 + 8LL * v48);
                  v49 = 1;
                  v36 = *(_DWORD *)(a3 + 16) + 1;
                  v50 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (v63.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != v50 )
                  {
                    while ( v50 != -8 )
                    {
                      if ( v50 == -16 && !v44 )
                        v44 = v29;
                      v48 = v46 & (v49 + v48);
                      v29 = (_QWORD *)(v47 + 8LL * v48);
                      v50 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (v63.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == v50 )
                        goto LABEL_65;
                      ++v49;
                    }
                    goto LABEL_77;
                  }
                }
LABEL_65:
                *(_DWORD *)(a3 + 16) = v36;
                if ( (*v29 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
                  --*(_DWORD *)(a3 + 20);
                *v29 = v63.m128i_i64[0];
                v37 = *(char **)(a3 + 40);
                if ( v37 == *(char **)(a3 + 48) )
                {
                  sub_142E0C0((char **)(a3 + 32), v37, &v63);
                }
                else
                {
                  if ( v37 )
                  {
                    *(_QWORD *)v37 = v63.m128i_i64[0];
                    v37 = *(char **)(a3 + 40);
                  }
                  *(_QWORD *)(a3 + 40) = v37 + 8;
                }
              }
            }
          }
        }
LABEL_41:
        v16 += 3;
      }
      while ( v7 != v16 );
    }
LABEL_3:
    v4 = v67;
    v5 = v68;
  }
  if ( v4 != v69 )
    _libc_free((unsigned __int64)v4);
  return v51;
}
