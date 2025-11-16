// Function: sub_242E400
// Address: 0x242e400
//
void __fastcall sub_242E400(__int64 a1, char *a2, __int64 a3, __int64 a4, int a5, int a6, int a7)
{
  __int64 v7; // r15
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r13d
  __int64 v11; // r14
  int v12; // r11d
  _QWORD *v13; // rdi
  __int64 v14; // r9
  _QWORD *v15; // rax
  void *v16; // r8
  void *v17; // rbx
  int v18; // r9d
  __int64 v19; // rsi
  unsigned int v20; // edx
  int v21; // eax
  __int64 v22; // rcx
  int v23; // r11d
  _QWORD *v24; // r10
  int v25; // eax
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // rax
  char *v29; // rdx
  unsigned __int8 *v30; // rax
  size_t v31; // rdx
  void **v32; // r12
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rax
  char *v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  int v39; // esi
  unsigned __int64 v40; // rbx
  int v41; // eax
  __int64 v42; // r13
  unsigned __int64 v43; // rdi
  __int64 v44; // r13
  __int64 v45; // r15
  _QWORD *v46; // r14
  unsigned __int64 v47; // rdi
  __int64 v48; // rbx
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rdi
  int v51; // ebx
  int v52; // r9d
  __int64 v53; // rsi
  int v54; // r11d
  unsigned int v55; // edx
  __int64 v56; // rcx
  __int64 v57; // [rsp+0h] [rbp-1B0h]
  __int64 v58; // [rsp+8h] [rbp-1A8h]
  __int64 v59; // [rsp+10h] [rbp-1A0h]
  __int64 v60; // [rsp+20h] [rbp-190h]
  __int64 v61; // [rsp+28h] [rbp-188h]
  unsigned __int64 v63; // [rsp+38h] [rbp-178h]
  unsigned __int64 v64; // [rsp+40h] [rbp-170h]
  unsigned int v65; // [rsp+40h] [rbp-170h]
  __int64 v67; // [rsp+60h] [rbp-150h]
  int v68; // [rsp+68h] [rbp-148h]
  size_t v69; // [rsp+68h] [rbp-148h]
  unsigned __int64 v70; // [rsp+78h] [rbp-138h] BYREF
  char *v71; // [rsp+80h] [rbp-130h] BYREF
  __int64 v72; // [rsp+88h] [rbp-128h]
  _QWORD v73[2]; // [rsp+90h] [rbp-120h] BYREF
  char v74; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v75; // [rsp+E0h] [rbp-D0h]
  __int64 v76; // [rsp+E8h] [rbp-C8h]
  __int64 v77; // [rsp+F0h] [rbp-C0h]
  void *v78; // [rsp+100h] [rbp-B0h] BYREF
  char *v79; // [rsp+108h] [rbp-A8h]
  __int64 v80; // [rsp+110h] [rbp-A0h]
  char *v81; // [rsp+118h] [rbp-98h]
  void *dest; // [rsp+120h] [rbp-90h]
  _QWORD v83[17]; // [rsp+128h] [rbp-88h] BYREF

  v7 = a1;
  *(_QWORD *)(a1 + 8) = a4;
  *(_DWORD *)(a1 + 28) = a7;
  v60 = a1 + 32;
  v61 = a1 + 80;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 16) = a5;
  *(_DWORD *)(a1 + 20) = a6;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = a2;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 104) = 0x400000000LL;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0xC000000000LL;
  *(_QWORD *)(a1 + 200) = a2;
  *(_DWORD *)(a1 + 208) = 1;
  *(_QWORD *)(a1 + 216) = a1 + 232;
  *(_QWORD *)(a1 + 224) = 0x400000000LL;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0xC000000000LL;
  v67 = a3 + 72;
  if ( *(_QWORD *)(a3 + 80) != a3 + 72 )
  {
    v8 = 0;
    v9 = 0;
    v10 = 2;
    v11 = *(_QWORD *)(a3 + 80);
    while ( 1 )
    {
      v17 = (void *)(v11 - 24);
      v83[8] = 0;
      v68 = v10 + 1;
      if ( !v11 )
        v17 = 0;
      LODWORD(v72) = v10;
      v73[0] = &v74;
      v71 = a2;
      v79 = a2;
      v73[1] = 0x400000000LL;
      v75 = 0;
      v76 = 0;
      v77 = 0xC000000000LL;
      v78 = v17;
      LODWORD(v80) = v10;
      v81 = (char *)v83;
      dest = (void *)0x400000000LL;
      v83[9] = 0;
      v83[10] = 0xC000000000LL;
      if ( !v8 )
        break;
      v12 = 1;
      v13 = 0;
      v14 = (v8 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v15 = (_QWORD *)(v9 + 16 * v14);
      v16 = (void *)*v15;
      if ( v17 == (void *)*v15 )
        goto LABEL_4;
      while ( v16 != (void *)-4096LL )
      {
        if ( v13 || v16 != (void *)-8192LL )
          v15 = v13;
        v14 = (v8 - 1) & (v12 + (_DWORD)v14);
        v16 = *(void **)(v9 + 16LL * (unsigned int)v14);
        if ( v17 == v16 )
          goto LABEL_4;
        ++v12;
        v13 = v15;
        v15 = (_QWORD *)(v9 + 16LL * (unsigned int)v14);
      }
      if ( !v13 )
        v13 = v15;
      v25 = *(_DWORD *)(v7 + 48);
      ++*(_QWORD *)(v7 + 32);
      v21 = v25 + 1;
      if ( 4 * v21 >= 3 * v8 )
        goto LABEL_10;
      if ( v8 - (v21 + *(_DWORD *)(v7 + 52)) <= v8 >> 3 )
      {
        v65 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
        sub_B23080(v60, v8);
        v52 = *(_DWORD *)(v7 + 56);
        if ( !v52 )
        {
LABEL_82:
          ++*(_DWORD *)(v7 + 48);
          BUG();
        }
        v14 = (unsigned int)(v52 - 1);
        v53 = *(_QWORD *)(v7 + 40);
        v24 = 0;
        v54 = 1;
        v21 = *(_DWORD *)(v7 + 48) + 1;
        v55 = v14 & v65;
        v13 = (_QWORD *)(v53 + 16LL * ((unsigned int)v14 & v65));
        v56 = *v13;
        if ( v17 != (void *)*v13 )
        {
          while ( v56 != -4096 )
          {
            if ( !v24 && v56 == -8192 )
              v24 = v13;
            v55 = v14 & (v54 + v55);
            v13 = (_QWORD *)(v53 + 16LL * v55);
            v56 = *v13;
            if ( v17 == (void *)*v13 )
              goto LABEL_26;
            ++v54;
          }
          goto LABEL_14;
        }
      }
LABEL_26:
      *(_DWORD *)(v7 + 48) = v21;
      if ( *v13 != -4096 )
        --*(_DWORD *)(v7 + 52);
      *((_DWORD *)v13 + 2) = 0;
      *v13 = v17;
      *((_DWORD *)v13 + 2) = *(_DWORD *)(v7 + 72);
      v26 = *(unsigned int *)(v7 + 72);
      v27 = v26;
      if ( *(_DWORD *)(v7 + 76) <= (unsigned int)v26 )
      {
        v59 = sub_C8D7D0(v7 + 64, v61, 0, 0x80u, &v70, v14);
        v33 = (unsigned __int64)*(unsigned int *)(v7 + 72) << 7;
        v34 = v33 + v59;
        if ( v33 + v59 )
        {
          *(_QWORD *)v34 = v17;
          v35 = v79;
          *(_QWORD *)(v34 + 32) = 0x400000000LL;
          *(_QWORD *)(v34 + 8) = v35;
          *(_DWORD *)(v34 + 16) = v10;
          *(_QWORD *)(v34 + 24) = v34 + 40;
          *(_QWORD *)(v34 + 104) = 0;
          *(_QWORD *)(v34 + 112) = 0;
          *(_QWORD *)(v34 + 120) = 0xC000000000LL;
          v33 = (unsigned __int64)*(unsigned int *)(v7 + 72) << 7;
        }
        v36 = *(_QWORD *)(v7 + 64);
        v63 = v36 + v33;
        if ( v36 != v36 + v33 )
        {
          v37 = v59;
          v38 = v59 + v33;
          do
          {
            if ( v37 )
            {
              *(_QWORD *)v37 = *(_QWORD *)v36;
              *(_QWORD *)(v37 + 8) = *(_QWORD *)(v36 + 8);
              v39 = *(_DWORD *)(v36 + 16);
              *(_DWORD *)(v37 + 32) = 0;
              *(_DWORD *)(v37 + 16) = v39;
              *(_QWORD *)(v37 + 24) = v37 + 40;
              *(_DWORD *)(v37 + 36) = 4;
              *(_QWORD *)(v37 + 104) = 0;
              *(_DWORD *)(v37 + 112) = 0;
              *(_DWORD *)(v37 + 116) = 0;
              *(_DWORD *)(v37 + 120) = 0;
              *(_DWORD *)(v37 + 124) = 192;
            }
            v37 += 128;
            v36 += 128;
          }
          while ( v38 != v37 );
          v63 = *(_QWORD *)(v7 + 64);
          v40 = v63 + ((unsigned __int64)*(unsigned int *)(v7 + 72) << 7);
          if ( v63 != v40 )
          {
            v58 = v11;
            v57 = v7;
            do
            {
              v41 = *(_DWORD *)(v40 - 12);
              v40 -= 128LL;
              if ( v41 && (v42 = *(unsigned int *)(v40 + 112), (_DWORD)v42) )
              {
                v64 = v40;
                v43 = *(_QWORD *)(v40 + 104);
                v44 = 8 * v42;
                v45 = 0;
                do
                {
                  v46 = *(_QWORD **)(v43 + v45);
                  if ( v46 != (_QWORD *)-8LL && v46 )
                  {
                    v47 = v46[6];
                    v48 = *v46 + 193LL;
                    if ( (_QWORD *)v47 != v46 + 8 )
                      _libc_free(v47);
                    v49 = v46[2];
                    if ( (_QWORD *)v49 != v46 + 4 )
                      j_j___libc_free_0(v49);
                    sub_C7D6A0((__int64)v46, v48, 8);
                    v43 = *(_QWORD *)(v64 + 104);
                  }
                  v45 += 8;
                }
                while ( v44 != v45 );
                v40 = v64;
              }
              else
              {
                v43 = *(_QWORD *)(v40 + 104);
              }
              _libc_free(v43);
              v50 = *(_QWORD *)(v40 + 24);
              if ( v50 != v40 + 40 )
                _libc_free(v50);
            }
            while ( v63 != v40 );
            v7 = v57;
            v11 = v58;
            v63 = *(_QWORD *)(v57 + 64);
          }
        }
        v51 = v70;
        if ( v61 != v63 )
          _libc_free(v63);
        ++*(_DWORD *)(v7 + 72);
        *(_DWORD *)(v7 + 76) = v51;
        *(_QWORD *)(v7 + 64) = v59;
LABEL_4:
        v11 = *(_QWORD *)(v11 + 8);
        if ( v67 == v11 )
          goto LABEL_32;
        goto LABEL_5;
      }
      v28 = *(_QWORD *)(v7 + 64) + (v26 << 7);
      if ( v28 )
      {
        *(_QWORD *)v28 = v17;
        v29 = v79;
        *(_QWORD *)(v28 + 32) = 0x400000000LL;
        *(_QWORD *)(v28 + 8) = v29;
        *(_DWORD *)(v28 + 16) = v10;
        *(_QWORD *)(v28 + 24) = v28 + 40;
        *(_QWORD *)(v28 + 104) = 0;
        *(_QWORD *)(v28 + 112) = 0;
        *(_QWORD *)(v28 + 120) = 0xC000000000LL;
        v27 = *(_DWORD *)(v7 + 72);
      }
      *(_DWORD *)(v7 + 72) = v27 + 1;
      v11 = *(_QWORD *)(v11 + 8);
      if ( v67 == v11 )
        goto LABEL_32;
LABEL_5:
      v9 = *(_QWORD *)(v7 + 40);
      v8 = *(_DWORD *)(v7 + 56);
      v10 = v68;
    }
    ++*(_QWORD *)(v7 + 32);
LABEL_10:
    sub_B23080(v60, 2 * v8);
    v18 = *(_DWORD *)(v7 + 56);
    if ( !v18 )
      goto LABEL_82;
    v14 = (unsigned int)(v18 - 1);
    v19 = *(_QWORD *)(v7 + 40);
    v20 = v14 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v21 = *(_DWORD *)(v7 + 48) + 1;
    v13 = (_QWORD *)(v19 + 16LL * v20);
    v22 = *v13;
    if ( v17 != (void *)*v13 )
    {
      v23 = 1;
      v24 = 0;
      while ( v22 != -4096 )
      {
        if ( !v24 && v22 == -8192 )
          v24 = v13;
        v20 = v14 & (v23 + v20);
        v13 = (_QWORD *)(v19 + 16LL * v20);
        v22 = *v13;
        if ( v17 == (void *)*v13 )
          goto LABEL_26;
        ++v23;
      }
LABEL_14:
      if ( v24 )
        v13 = v24;
      goto LABEL_26;
    }
    goto LABEL_26;
  }
LABEL_32:
  v83[0] = 0x100000000LL;
  v83[1] = &v71;
  v78 = &unk_49DD210;
  v71 = (char *)v73;
  v72 = 0;
  LOBYTE(v73[0]) = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  dest = 0;
  sub_CB5980((__int64)&v78, 0, 0, 0);
  v30 = (unsigned __int8 *)sub_2426330(a4);
  if ( v31 > v81 - (_BYTE *)dest )
  {
    v32 = (void **)sub_CB6200((__int64)&v78, v30, v31);
  }
  else
  {
    v32 = &v78;
    if ( v31 )
    {
      v69 = v31;
      memcpy(dest, v30, v31);
      dest = (char *)dest + v69;
    }
  }
  sub_CB59D0((__int64)v32, *(unsigned int *)(a4 + 16));
  *(_DWORD *)(v7 + 24) = sub_B3B940(v71, &v71[v72]);
  v78 = &unk_49DD210;
  sub_CB5840((__int64)&v78);
  if ( v71 != (char *)v73 )
    j_j___libc_free_0((unsigned __int64)v71);
}
