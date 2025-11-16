// Function: sub_24A9E20
// Address: 0x24a9e20
//
void __fastcall sub_24A9E20(__int64 **a1, __int64 *a2)
{
  __int64 **v2; // r15
  __int64 *v3; // rdx
  int v4; // eax
  unsigned __int64 *v5; // rax
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  int v10; // eax
  __int64 *v11; // rdi
  unsigned int v12; // esi
  unsigned __int64 v13; // r10
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r10
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  int v22; // eax
  _BYTE *v23; // rdi
  int v24; // esi
  unsigned __int64 v25; // r10
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r10
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdx
  char *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  char v36; // r13
  __int64 v37; // rax
  __int64 v38; // r9
  unsigned __int8 v39; // dl
  __int64 v40; // rcx
  __int64 v41; // r8
  _BYTE **v42; // rbx
  _BYTE **v43; // r14
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  _BYTE *v46; // r13
  size_t v47; // r15
  const void *v48; // rax
  __int64 v49; // rdx
  __int64 *v50; // rdi
  size_t v51; // rax
  __int64 v52; // rbx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 v57; // rax
  char v58; // [rsp+38h] [rbp-248h]
  __int64 v59; // [rsp+40h] [rbp-240h]
  __int64 *v61; // [rsp+50h] [rbp-230h]
  __m128i v62; // [rsp+60h] [rbp-220h] BYREF
  __int64 v63; // [rsp+70h] [rbp-210h] BYREF
  unsigned __int64 v64[2]; // [rsp+80h] [rbp-200h] BYREF
  __int64 v65; // [rsp+90h] [rbp-1F0h] BYREF
  unsigned __int64 v66[2]; // [rsp+A0h] [rbp-1E0h] BYREF
  _WORD v67[8]; // [rsp+B0h] [rbp-1D0h] BYREF
  __m128i v68; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v69; // [rsp+D0h] [rbp-1B0h] BYREF
  __int64 v70[2]; // [rsp+E0h] [rbp-1A0h] BYREF
  _BYTE v71[16]; // [rsp+F0h] [rbp-190h] BYREF
  __m128i v72; // [rsp+100h] [rbp-180h] BYREF
  __int64 v73; // [rsp+110h] [rbp-170h] BYREF
  unsigned __int64 v74[2]; // [rsp+120h] [rbp-160h] BYREF
  char v75[16]; // [rsp+130h] [rbp-150h] BYREF
  __m128i v76; // [rsp+140h] [rbp-140h] BYREF
  __int64 v77; // [rsp+150h] [rbp-130h] BYREF
  _QWORD *v78; // [rsp+160h] [rbp-120h] BYREF
  int v79; // [rsp+168h] [rbp-118h]
  _QWORD v80[2]; // [rsp+170h] [rbp-110h] BYREF
  __m128i v81; // [rsp+180h] [rbp-100h] BYREF
  __int64 v82; // [rsp+190h] [rbp-F0h] BYREF
  unsigned __int64 v83[2]; // [rsp+1A0h] [rbp-E0h] BYREF
  _QWORD v84[2]; // [rsp+1B0h] [rbp-D0h] BYREF
  __m128i v85; // [rsp+1C0h] [rbp-C0h] BYREF
  __int64 v86; // [rsp+1D0h] [rbp-B0h] BYREF
  __int64 *v87; // [rsp+1E0h] [rbp-A0h] BYREF
  int v88; // [rsp+1E8h] [rbp-98h]
  _QWORD v89[2]; // [rsp+1F0h] [rbp-90h] BYREF
  char s[16]; // [rsp+200h] [rbp-80h] BYREF
  _QWORD v91[2]; // [rsp+210h] [rbp-70h] BYREF
  __m128i *v92; // [rsp+220h] [rbp-60h] BYREF
  __int64 v93; // [rsp+228h] [rbp-58h]
  _QWORD v94[2]; // [rsp+230h] [rbp-50h] BYREF
  __int16 v95; // [rsp+240h] [rbp-40h]

  v2 = a1;
  v3 = *a1;
  v61 = *(__int64 **)(*a1)[1];
  v4 = *((_DWORD *)a2 + 2);
  if ( v4 == 13 )
  {
    v36 = unk_4FEBF28 ^ 1;
    goto LABEL_86;
  }
  if ( v4 != 15 && v4 != 9 )
    goto LABEL_4;
  v35 = *v3;
  v36 = unk_4FEBE48;
  v59 = *v3;
  if ( !unk_4FEBE48 )
  {
    v36 = qword_4FEBCE0[17];
    if ( v36 )
    {
      if ( !*(_QWORD *)(v35 + 48) )
        v36 = (*(_BYTE *)(v35 + 32) & 0xF) == 1 || (*(_BYTE *)(v35 + 32) & 0xF) == 4;
    }
  }
  strcpy((char *)v91, "mismatch");
  v92 = (__m128i *)v94;
  v93 = 0x200000000LL;
  *(__m128i *)s = _mm_load_si128((const __m128i *)&xmmword_4386FA0);
  if ( (*(_BYTE *)(v59 + 7) & 0x20) == 0
    || (v37 = sub_B91C10(v59, 30)) == 0
    || ((v39 = *(_BYTE *)(v37 - 16), (v39 & 2) != 0)
      ? (v41 = *(_QWORD *)(v37 - 32), v40 = *(unsigned int *)(v37 - 24))
      : (v40 = (*(_WORD *)(v37 - 16) >> 6) & 0xF, v41 = v37 - 16 - 8LL * ((v39 >> 2) & 0xF)),
        v42 = (_BYTE **)(v41 + 8 * v40),
        v42 == (_BYTE **)v41) )
  {
LABEL_100:
    v87 = v61;
    v51 = strlen(s);
    v52 = sub_B8C130(&v87, (__int64)s, v51);
    v55 = (unsigned int)v93;
    v56 = (unsigned int)v93 + 1LL;
    if ( v56 > HIDWORD(v93) )
    {
      sub_C8D5F0((__int64)&v92, v94, v56, 8u, v53, v54);
      v55 = (unsigned int)v93;
    }
    v92->m128i_i64[v55] = v52;
    LODWORD(v93) = v93 + 1;
    v57 = sub_B9C770(v61, v92->m128i_i64, (__int64 *)(unsigned int)v93, 0, 1);
    sub_B99110(v59, 30, v57);
    v50 = (__int64 *)v92;
    if ( v92 != (__m128i *)v94 )
      goto LABEL_79;
    goto LABEL_86;
  }
  v58 = v36;
  v43 = (_BYTE **)v41;
  while ( 1 )
  {
    v46 = *v43;
    if ( **v43 )
    {
      v44 = (unsigned int)v93;
      v45 = (unsigned int)v93 + 1LL;
      if ( v45 > HIDWORD(v93) )
        goto LABEL_93;
      goto LABEL_73;
    }
    v47 = strlen(s);
    v48 = (const void *)sub_B91420((__int64)v46);
    if ( v47 == v49 && (!v47 || !memcmp(v48, s, v47)) )
      break;
    v44 = (unsigned int)v93;
    v46 = *v43;
    v45 = (unsigned int)v93 + 1LL;
    if ( v45 > HIDWORD(v93) )
    {
LABEL_93:
      sub_C8D5F0((__int64)&v92, v94, v45, 8u, v41, v38);
      v44 = (unsigned int)v93;
    }
LABEL_73:
    ++v43;
    v92->m128i_i64[v44] = (__int64)v46;
    LODWORD(v93) = v93 + 1;
    if ( v42 == v43 )
    {
      v36 = v58;
      v2 = a1;
      goto LABEL_100;
    }
  }
  v36 = v58;
  v2 = a1;
  v50 = (__int64 *)v92;
  if ( v92 == (__m128i *)v94 )
    goto LABEL_86;
LABEL_79:
  _libc_free((unsigned __int64)v50);
LABEL_86:
  if ( !v36 )
  {
LABEL_4:
    *(_QWORD *)s = 16;
    v92 = (__m128i *)v94;
    v92 = (__m128i *)sub_22409D0((__int64)&v92, (unsigned __int64 *)s, 0);
    v94[0] = *(_QWORD *)s;
    *v92 = _mm_load_si128((const __m128i *)&xmmword_4386FB0);
    v93 = *(_QWORD *)s;
    v92->m128i_i8[*(_QWORD *)s] = 0;
    v5 = (unsigned __int64 *)v2[1];
    v6 = *v5;
    if ( *v5 > 9 )
    {
      if ( v6 <= 0x63 )
      {
        v87 = v89;
        sub_2240A50((__int64 *)&v87, 2u, 0);
        v11 = v87;
      }
      else
      {
        if ( v6 <= 0x3E7 )
        {
          v8 = 3;
        }
        else if ( v6 <= 0x270F )
        {
          v8 = 4;
        }
        else
        {
          v7 = *v5;
          LODWORD(v8) = 1;
          while ( 1 )
          {
            v9 = v7;
            v10 = v8;
            v8 = (unsigned int)(v8 + 4);
            v7 /= 0x2710u;
            if ( v9 <= 0x1869F )
              break;
            if ( v9 <= 0xF423F )
            {
              v8 = (unsigned int)(v10 + 5);
              v87 = v89;
              goto LABEL_14;
            }
            if ( v9 <= (unsigned __int64)&loc_98967F )
            {
              v8 = (unsigned int)(v10 + 6);
              break;
            }
            if ( v9 <= 0x5F5E0FF )
            {
              v8 = (unsigned int)(v10 + 7);
              break;
            }
          }
        }
        v87 = v89;
LABEL_14:
        sub_2240A50((__int64 *)&v87, v8, 0);
        v11 = v87;
        v12 = v88 - 1;
        do
        {
          v13 = v6;
          v14 = 5 * (v6 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v6 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
          v15 = v6;
          v6 /= 0x64u;
          v16 = v13 - 4 * v14;
          *((_BYTE *)v11 + v12) = a00010203040506_0[2 * v16 + 1];
          v17 = v12 - 1;
          v12 -= 2;
          *((_BYTE *)v11 + v17) = a00010203040506_0[2 * v16];
        }
        while ( v15 > 0x270F );
        if ( v15 <= 0x3E7 )
          goto LABEL_17;
      }
      *((_BYTE *)v11 + 1) = a00010203040506_0[2 * v6 + 1];
      *(_BYTE *)v11 = a00010203040506_0[2 * v6];
LABEL_18:
      v84[0] = 0x206F7420707520LL;
      v83[0] = (unsigned __int64)v84;
      v18 = (*v2)[29];
      v83[1] = 7;
      if ( v18 > 9 )
      {
        if ( v18 <= 0x63 )
        {
          v78 = v80;
          sub_2240A50((__int64 *)&v78, 2u, 0);
          v23 = v78;
        }
        else
        {
          if ( v18 <= 0x3E7 )
          {
            v20 = 3;
          }
          else if ( v18 <= 0x270F )
          {
            v20 = 4;
          }
          else
          {
            v19 = v18;
            LODWORD(v20) = 1;
            while ( 1 )
            {
              v21 = v19;
              v22 = v20;
              v20 = (unsigned int)(v20 + 4);
              v19 /= 0x2710u;
              if ( v21 <= 0x1869F )
                break;
              if ( v21 <= 0xF423F )
              {
                v20 = (unsigned int)(v22 + 5);
                v78 = v80;
                goto LABEL_28;
              }
              if ( v21 <= (unsigned __int64)&loc_98967F )
              {
                v20 = (unsigned int)(v22 + 6);
                break;
              }
              if ( v21 <= 0x5F5E0FF )
              {
                v20 = (unsigned int)(v22 + 7);
                break;
              }
            }
          }
          v78 = v80;
LABEL_28:
          sub_2240A50((__int64 *)&v78, v20, 0);
          v23 = v78;
          v24 = v79 - 1;
          do
          {
            v25 = v18;
            v26 = 5
                * (v18 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v18 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
            v27 = v18;
            v18 /= 0x64u;
            v28 = v25 - 4 * v26;
            v23[v24] = a00010203040506_0[2 * v28 + 1];
            v29 = (unsigned int)(v24 - 1);
            v24 -= 2;
            v23[v29] = a00010203040506_0[2 * v28];
          }
          while ( v27 > 0x270F );
          if ( v27 <= 0x3E7 )
            goto LABEL_31;
        }
        v23[1] = a00010203040506_0[2 * v18 + 1];
        *v23 = a00010203040506_0[2 * v18];
        goto LABEL_32;
      }
      v78 = v80;
      sub_2240A50((__int64 *)&v78, 1u, 0);
      v23 = v78;
LABEL_31:
      *v23 = v18 + 48;
LABEL_32:
      v74[0] = (unsigned __int64)v75;
      strcpy(v75, " Hash = ");
      v30 = *v2;
      v74[1] = 8;
      v32 = (char *)sub_BD5D20(*v30);
      if ( v32 )
      {
        v70[0] = (__int64)v71;
        sub_24A2F70(v70, v32, (__int64)&v32[v31]);
      }
      else
      {
        v71[0] = 0;
        v70[0] = (__int64)v71;
        v70[1] = 0;
      }
      v33 = *a2;
      v67[0] = 32;
      v66[0] = (unsigned __int64)v67;
      v66[1] = 1;
      (*(void (__fastcall **)(unsigned __int64 *))(v33 + 24))(v64);
      sub_8FD5D0(&v68, (__int64)v64, v66);
      sub_8FD5D0(&v72, (__int64)&v68, v70);
      sub_8FD5D0(&v76, (__int64)&v72, v74);
      sub_8FD5D0(&v81, (__int64)&v76, &v78);
      sub_8FD5D0(&v85, (__int64)&v81, v83);
      sub_8FD5D0((__m128i *)s, (__int64)&v85, &v87);
      sub_8FD5D0(&v62, (__int64)s, &v92);
      if ( *(_QWORD **)s != v91 )
        j_j___libc_free_0(*(unsigned __int64 *)s);
      if ( (__int64 *)v85.m128i_i64[0] != &v86 )
        j_j___libc_free_0(v85.m128i_u64[0]);
      if ( (__int64 *)v81.m128i_i64[0] != &v82 )
        j_j___libc_free_0(v81.m128i_u64[0]);
      if ( (__int64 *)v76.m128i_i64[0] != &v77 )
        j_j___libc_free_0(v76.m128i_u64[0]);
      if ( (__int64 *)v72.m128i_i64[0] != &v73 )
        j_j___libc_free_0(v72.m128i_u64[0]);
      if ( (__int64 *)v68.m128i_i64[0] != &v69 )
        j_j___libc_free_0(v68.m128i_u64[0]);
      if ( (__int64 *)v64[0] != &v65 )
        j_j___libc_free_0(v64[0]);
      if ( (_WORD *)v66[0] != v67 )
        j_j___libc_free_0(v66[0]);
      if ( (_BYTE *)v70[0] != v71 )
        j_j___libc_free_0(v70[0]);
      if ( (char *)v74[0] != v75 )
        j_j___libc_free_0(v74[0]);
      if ( v78 != v80 )
        j_j___libc_free_0((unsigned __int64)v78);
      if ( (_QWORD *)v83[0] != v84 )
        j_j___libc_free_0(v83[0]);
      if ( v87 != v89 )
        j_j___libc_free_0((unsigned __int64)v87);
      if ( v92 != (__m128i *)v94 )
        j_j___libc_free_0((unsigned __int64)v92);
      v92 = &v62;
      v95 = 260;
      v34 = *(_QWORD *)((*v2)[1] + 168);
      *(_QWORD *)&s[8] = 0x100000017LL;
      *(_QWORD *)s = &unk_49D9CA8;
      v91[0] = v34;
      v91[1] = &v92;
      sub_B6EB20((__int64)v61, (__int64)s);
      if ( (__int64 *)v62.m128i_i64[0] != &v63 )
        j_j___libc_free_0(v62.m128i_u64[0]);
      return;
    }
    v87 = v89;
    sub_2240A50((__int64 *)&v87, 1u, 0);
    v11 = v87;
LABEL_17:
    *(_BYTE *)v11 = v6 + 48;
    goto LABEL_18;
  }
}
