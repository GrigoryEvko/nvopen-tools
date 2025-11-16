// Function: sub_31C63D0
// Address: 0x31c63d0
//
__int64 __fastcall sub_31C63D0(__int64 a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // r15
  char v5; // al
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned int v8; // r14d
  __int64 v9; // r13
  _BYTE *v10; // r12
  _BYTE *v11; // rdx
  _BYTE *v12; // rdx
  __int64 v13; // rax
  __int64 v15; // rax
  _BYTE *v16; // rdx
  unsigned __int8 v17; // al
  __int64 *v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // r12
  char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r8
  unsigned int v24; // r13d
  char v25; // al
  __int64 *v26; // rdi
  __int64 v27; // r12
  unsigned int v28; // eax
  __int64 v29; // rax
  int v30; // ecx
  _BYTE *v31; // rdx
  unsigned __int64 v32; // rdx
  const char *v33; // r10
  size_t v34; // r8
  size_t v35; // rax
  _QWORD *v36; // rdx
  size_t v37; // rdx
  const char *v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 *v41; // r12
  size_t v42; // rdx
  _QWORD *v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rcx
  __int64 v46; // rax
  _QWORD *v47; // rdi
  char *v48; // rax
  size_t v49; // rdx
  size_t v50; // rdx
  size_t v51; // [rsp+8h] [rbp-258h]
  const char *v52; // [rsp+10h] [rbp-250h]
  __int64 v53; // [rsp+18h] [rbp-248h]
  int v54; // [rsp+60h] [rbp-200h]
  size_t v56; // [rsp+78h] [rbp-1E8h] BYREF
  _QWORD *v57; // [rsp+80h] [rbp-1E0h] BYREF
  size_t v58; // [rsp+88h] [rbp-1D8h]
  _QWORD src[2]; // [rsp+90h] [rbp-1D0h] BYREF
  void *s1; // [rsp+A0h] [rbp-1C0h] BYREF
  size_t n; // [rsp+A8h] [rbp-1B8h]
  __int64 (__fastcall **v62)(); // [rsp+B0h] [rbp-1B0h] BYREF
  _QWORD v63[3]; // [rsp+B8h] [rbp-1A8h] BYREF
  unsigned __int64 v64; // [rsp+D0h] [rbp-190h]
  _BYTE *v65; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v66; // [rsp+E0h] [rbp-180h]
  __int64 v67; // [rsp+E8h] [rbp-178h]
  volatile signed __int32 *v68; // [rsp+F0h] [rbp-170h] BYREF
  int v69; // [rsp+F8h] [rbp-168h]
  unsigned __int64 v70[2]; // [rsp+100h] [rbp-160h] BYREF
  _BYTE v71[16]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v72[28]; // [rsp+120h] [rbp-140h] BYREF
  __int16 v73; // [rsp+200h] [rbp-60h]
  __int64 v74; // [rsp+208h] [rbp-58h]
  __int64 v75; // [rsp+210h] [rbp-50h]
  __int64 v76; // [rsp+218h] [rbp-48h]
  __int64 v77; // [rsp+220h] [rbp-40h]

  v3 = a2;
  v5 = *a2;
  if ( *a2 <= 0x1Cu )
  {
LABEL_5:
    v8 = sub_CEFB70(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 40LL));
    if ( !(_BYTE)v8 )
      return 0;
    v9 = *(_QWORD *)(a1 + 256);
    if ( (*(_BYTE *)(v9 + 2) & 1) != 0 )
    {
      sub_B2C6D0(*(_QWORD *)(a1 + 256), (__int64)a2, v6, v7);
      v10 = *(_BYTE **)(v9 + 96);
      v9 = *(_QWORD *)(a1 + 256);
      if ( (*(_BYTE *)(v9 + 2) & 1) != 0 )
        sub_B2C6D0(*(_QWORD *)(a1 + 256), (__int64)a2, v39, v40);
      v11 = *(_BYTE **)(v9 + 96);
    }
    else
    {
      v10 = *(_BYTE **)(v9 + 96);
      v11 = v10;
    }
    v12 = &v11[40 * *(_QWORD *)(v9 + 104)];
    if ( v12 == v10 )
      return 0;
    if ( v10 != v3 )
    {
      LODWORD(v13) = 0;
      while ( 1 )
      {
        v10 += 40;
        v13 = (unsigned int)(v13 + 1);
        if ( v10 == v12 )
          return 0;
        if ( v10 == v3 )
        {
          v53 = v13;
          goto LABEL_47;
        }
      }
    }
    v53 = 0;
LABEL_47:
    sub_222DF20((__int64)v72);
    v72[27] = 0;
    v74 = 0;
    v75 = 0;
    v72[0] = off_4A06798;
    s1 = qword_4A072D8;
    v73 = 0;
    v76 = 0;
    v77 = 0;
    *(void **)((char *)&s1 + qword_4A072D8[-3]) = &unk_4A07300;
    n = 0;
    sub_222DD70((__int64)&s1 + *((_QWORD *)s1 - 3), 0);
    v62 = (__int64 (__fastcall **)())qword_4A07288;
    *(_QWORD *)((char *)&v63[-1] + qword_4A07288[-3]) = &unk_4A072B0;
    sub_222DD70((__int64)&v63[-1] + (_QWORD)*(v62 - 3), 0);
    s1 = qword_4A07328;
    *(void **)((char *)&s1 + qword_4A07328[-3]) = &unk_4A07378;
    s1 = off_4A073F0;
    v72[0] = off_4A07440;
    v62 = off_4A07418;
    v63[0] = off_4A07480;
    v63[1] = 0;
    v63[2] = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    sub_220A990(&v68);
    v69 = 24;
    v70[1] = 0;
    v63[0] = off_4A07080;
    v70[0] = (unsigned __int64)v71;
    v71[0] = 0;
    sub_222DD70((__int64)v72, (__int64)v63);
    v33 = sub_BD5D20(*(_QWORD *)(a1 + 256));
    v34 = v32;
    if ( !v33 )
    {
      LOBYTE(src[0]) = 0;
      v37 = 0;
      v57 = src;
      v38 = (const char *)src;
      v58 = 0;
LABEL_62:
      v41 = sub_223E0D0((__int64 *)&v62, v38, v37);
      sub_223E0D0(v41, "_param_", 7);
      sub_223E760(v41, v53);
      if ( v57 != src )
        j_j___libc_free_0((unsigned __int64)v57);
      v57 = src;
      v58 = 0;
      LOBYTE(src[0]) = 0;
      if ( v66 )
      {
        if ( v66 <= v64 )
          sub_2241130((unsigned __int64 *)&v57, 0, 0, v65, v64 - (_QWORD)v65);
        else
          sub_2241130((unsigned __int64 *)&v57, 0, 0, v65, v66 - (_QWORD)v65);
      }
      else
      {
        sub_2240AE0((unsigned __int64 *)&v57, v70);
      }
      v42 = v58;
      v43 = *(_QWORD **)a3;
      if ( v57 == src )
      {
        if ( v58 )
        {
          if ( v58 == 1 )
            *(_BYTE *)v43 = src[0];
          else
            memcpy(v43, src, v58);
          v42 = v58;
          v43 = *(_QWORD **)a3;
        }
        *(_QWORD *)(a3 + 8) = v42;
        *((_BYTE *)v43 + v42) = 0;
        v43 = v57;
        goto LABEL_71;
      }
      v44 = src[0];
      if ( v43 == (_QWORD *)(a3 + 16) )
      {
        *(_QWORD *)a3 = v57;
        *(_QWORD *)(a3 + 8) = v42;
        *(_QWORD *)(a3 + 16) = v44;
      }
      else
      {
        v45 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)a3 = v57;
        *(_QWORD *)(a3 + 8) = v42;
        *(_QWORD *)(a3 + 16) = v44;
        if ( v43 )
        {
          v57 = v43;
          src[0] = v45;
LABEL_71:
          v58 = 0;
          *(_BYTE *)v43 = 0;
          if ( v57 != src )
            j_j___libc_free_0((unsigned __int64)v57);
          s1 = off_4A073F0;
          v72[0] = off_4A07440;
          v62 = off_4A07418;
          v63[0] = off_4A07080;
          if ( (_BYTE *)v70[0] != v71 )
            j_j___libc_free_0(v70[0]);
          v63[0] = off_4A07480;
          sub_2209150(&v68);
          s1 = qword_4A07328;
          *(void **)((char *)&s1 + qword_4A07328[-3]) = &unk_4A07378;
          v62 = (__int64 (__fastcall **)())qword_4A07288;
          *(_QWORD *)((char *)&v63[-1] + qword_4A07288[-3]) = &unk_4A072B0;
          s1 = qword_4A072D8;
          *(void **)((char *)&s1 + qword_4A072D8[-3]) = &unk_4A07300;
          n = 0;
          v72[0] = off_4A06798;
          sub_222E050((__int64)v72);
          return v8;
        }
      }
      v57 = src;
      v43 = src;
      goto LABEL_71;
    }
    v56 = v32;
    v35 = v32;
    v57 = src;
    if ( v32 > 0xF )
    {
      v51 = v32;
      v52 = v33;
      v46 = sub_22409D0((__int64)&v57, &v56, 0);
      v33 = v52;
      v34 = v51;
      v57 = (_QWORD *)v46;
      v47 = (_QWORD *)v46;
      src[0] = v56;
    }
    else
    {
      if ( v32 == 1 )
      {
        LOBYTE(src[0]) = *v33;
        v36 = src;
LABEL_51:
        v58 = v35;
        *((_BYTE *)v36 + v35) = 0;
        v37 = v58;
        v38 = (const char *)v57;
        goto LABEL_62;
      }
      if ( !v32 )
      {
        v36 = src;
        goto LABEL_51;
      }
      v47 = src;
    }
    memcpy(v47, v33, v34);
    v35 = v56;
    v36 = v57;
    goto LABEL_51;
  }
  while ( v5 != 85 )
  {
    if ( v5 != 61 )
    {
      if ( v5 != 84 )
        goto LABEL_5;
      if ( (*((_DWORD *)v3 + 1) & 0x7FFFFFF) != 0 )
      {
        v8 = sub_31C63D0(a1, **((_QWORD **)v3 - 1), a3);
        if ( (_BYTE)v8 )
        {
          v24 = 1;
          v54 = *((_DWORD *)v3 + 1) & 0x7FFFFFF;
          if ( v54 == 1 )
            return v8;
          while ( 1 )
          {
            s1 = &v62;
            n = 0;
            LOBYTE(v62) = 0;
            v25 = sub_31C63D0(a1, *(_QWORD *)(*((_QWORD *)v3 - 1) + 32LL * v24), &s1);
            v26 = (__int64 *)s1;
            if ( !v25 )
              break;
            if ( n != *(_QWORD *)(a3 + 8) )
              break;
            if ( n )
            {
              v26 = (__int64 *)s1;
              if ( memcmp(s1, *(const void **)a3, n) )
                break;
            }
            if ( v26 != (__int64 *)&v62 )
              j_j___libc_free_0((unsigned __int64)v26);
            if ( ++v24 == v54 )
              return (unsigned __int8)v8;
          }
          if ( v26 != (__int64 *)&v62 )
            j_j___libc_free_0((unsigned __int64)v26);
        }
      }
      return 0;
    }
    v27 = *((_QWORD *)v3 - 4);
    if ( !v27 )
      BUG();
    if ( *(_BYTE *)v27 <= 3u )
    {
      v28 = sub_CE8830(*((_BYTE **)v3 - 4));
      if ( (_BYTE)v28 )
      {
        v8 = v28;
        v48 = (char *)sub_CEF7C0(v27);
        sub_2241130((unsigned __int64 *)a3, 0, *(_QWORD *)(a3 + 8), v48, v49);
        return v8;
      }
    }
    v29 = *(_QWORD *)(v27 + 16);
    if ( v29 )
    {
      v30 = 0;
      a2 = 0;
      do
      {
        while ( 1 )
        {
          v31 = *(_BYTE **)(v29 + 24);
          if ( *v31 == 62 )
            break;
          v29 = *(_QWORD *)(v29 + 8);
          if ( !v29 )
            goto LABEL_43;
        }
        v29 = *(_QWORD *)(v29 + 8);
        ++v30;
        a2 = v31;
      }
      while ( v29 );
LABEL_43:
      if ( v30 == 1 )
      {
        v3 = (_BYTE *)*((_QWORD *)a2 - 8);
        v5 = *v3;
        if ( *v3 > 0x1Cu )
          continue;
      }
    }
    goto LABEL_5;
  }
  v15 = *((_QWORD *)v3 - 4);
  if ( !v15 )
    goto LABEL_5;
  if ( *(_BYTE *)v15 )
    goto LABEL_5;
  a2 = (_BYTE *)*((_QWORD *)v3 + 10);
  if ( *(_BYTE **)(v15 + 24) != a2 )
    goto LABEL_5;
  if ( (*(_BYTE *)(v15 + 33) & 0x20) == 0 )
    goto LABEL_5;
  if ( *(_DWORD *)(v15 + 36) != 10578 )
    goto LABEL_5;
  v16 = *(_BYTE **)(*(_QWORD *)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)] + 24LL);
  if ( (unsigned __int8)(*v16 - 5) > 0x1Fu )
    goto LABEL_5;
  v17 = *(v16 - 16);
  v18 = (v17 & 2) != 0 ? (__int64 *)*((_QWORD *)v16 - 4) : (__int64 *)&v16[-8 * ((v17 >> 2) & 0xF) - 16];
  v19 = *v18;
  if ( *(_BYTE *)v19 != 1 )
    goto LABEL_5;
  v20 = *(_BYTE **)(v19 + 136);
  if ( *v20 != 3 )
    goto LABEL_5;
  if ( (unsigned __int8)sub_CE8750(*(_BYTE **)(v19 + 136)) )
  {
    v21 = (char *)sub_CEF7A0((__int64)v20);
    v23 = v22;
  }
  else
  {
    if ( (unsigned __int8)sub_CE87C0(v20) )
      v21 = (char *)sub_CEF7B0((__int64)v20);
    else
      v21 = (char *)sub_CEF7C0((__int64)v20);
    v23 = v50;
  }
  v8 = 1;
  sub_2241130((unsigned __int64 *)a3, 0, *(_QWORD *)(a3 + 8), v21, v23);
  return v8;
}
