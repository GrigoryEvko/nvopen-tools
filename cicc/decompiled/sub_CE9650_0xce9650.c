// Function: sub_CE9650
// Address: 0xce9650
//
__int64 __fastcall sub_CE9650(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // r13d
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // r15
  __int64 (__fastcall **v15)(); // rax
  int v16; // r8d
  char *v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r8
  char *v21; // r15
  _QWORD *v22; // rsi
  _BYTE *v23; // rdi
  size_t v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdi
  __int64 (__fastcall *v27)(__int64, unsigned int); // rax
  __int64 v28; // rdi
  __int64 (__fastcall *v29)(__int64, unsigned int); // rax
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 (__fastcall *v32)(__int64, unsigned int); // rax
  __int64 (__fastcall *v33)(__int64, unsigned int); // r8
  unsigned __int8 v34; // [rsp+Fh] [rbp-241h]
  int v35; // [rsp+58h] [rbp-1F8h]
  __int64 v36; // [rsp+58h] [rbp-1F8h]
  __int64 v37; // [rsp+58h] [rbp-1F8h]
  __int64 v38; // [rsp+58h] [rbp-1F8h]
  __int64 v39; // [rsp+58h] [rbp-1F8h]
  __int64 v40; // [rsp+68h] [rbp-1E8h] BYREF
  _QWORD *v41; // [rsp+70h] [rbp-1E0h] BYREF
  size_t n; // [rsp+78h] [rbp-1D8h]
  _QWORD src[2]; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 (__fastcall **v44)(); // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v45; // [rsp+98h] [rbp-1B8h]
  __int64 (__fastcall **v46)(); // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 (__fastcall **v47)(); // [rsp+A8h] [rbp-1A8h] BYREF
  __int64 v48; // [rsp+B0h] [rbp-1A0h]
  __int64 v49; // [rsp+B8h] [rbp-198h]
  unsigned __int64 v50; // [rsp+C0h] [rbp-190h]
  __int64 v51; // [rsp+C8h] [rbp-188h]
  unsigned __int64 v52; // [rsp+D0h] [rbp-180h]
  __int64 v53; // [rsp+D8h] [rbp-178h]
  char v54[8]; // [rsp+E0h] [rbp-170h] BYREF
  int v55; // [rsp+E8h] [rbp-168h]
  _QWORD v56[2]; // [rsp+F0h] [rbp-160h] BYREF
  _QWORD v57[2]; // [rsp+100h] [rbp-150h] BYREF
  _QWORD v58[28]; // [rsp+110h] [rbp-140h] BYREF
  __int16 v59; // [rsp+1F0h] [rbp-60h]
  __int64 v60; // [rsp+1F8h] [rbp-58h]
  __int64 v61; // [rsp+200h] [rbp-50h]
  __int64 v62; // [rsp+208h] [rbp-48h]
  __int64 v63; // [rsp+210h] [rbp-40h]

  v3 = sub_CE7BB0(a1, "unified.entry.id", 0x10u, &v40);
  if ( !(_BYTE)v3 )
    return v3;
  sub_222DF20(v58);
  v58[27] = 0;
  v60 = 0;
  v61 = 0;
  v58[0] = off_4A06798;
  v59 = 0;
  v62 = 0;
  v44 = (__int64 (__fastcall **)())qword_4A072D8;
  v63 = 0;
  *(__int64 (__fastcall ***)())((char *)&v44 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v45 = 0;
  sub_222DD70((char *)&v44 + (_QWORD)*(v44 - 3), 0);
  v46 = (__int64 (__fastcall **)())qword_4A07288;
  *(__int64 (__fastcall ***)())((char *)&v46 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
  sub_222DD70((char *)&v46 + (_QWORD)*(v46 - 3), 0);
  v44 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v44 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v48 = 0;
  v44 = off_4A073F0;
  v58[0] = off_4A07440;
  v46 = off_4A07418;
  v49 = 0;
  v47 = off_4A07480;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  sub_220A990(v54);
  v55 = 24;
  v47 = off_4A07080;
  v56[0] = v57;
  v56[1] = 0;
  LOBYTE(v57[0]) = 0;
  sub_222DD70(v58, &v47);
  v5 = (__int64)"0x";
  sub_223E0D0(&v46, "0x", 2);
  v7 = v40;
  v8 = (*(_BYTE *)(v40 - 16) & 2) != 0;
  if ( (*(_BYTE *)(v40 - 16) & 2) != 0 )
    v9 = *(unsigned int *)(v40 - 24);
  else
    v9 = (*(_WORD *)(v40 - 16) >> 6) & 0xF;
  if ( (_DWORD)v9 )
  {
    v10 = 0;
    v34 = v3;
    while ( 1 )
    {
      if ( (_BYTE)v8 )
      {
        v11 = *(_QWORD *)(v7 - 32);
      }
      else
      {
        v8 = 8LL * ((*(_BYTE *)(v7 - 16) >> 2) & 0xF);
        v5 = -16 - v8;
        v11 = -16 - v8 + v7;
      }
      v12 = *(_QWORD *)(v11 + 8 * v10);
      if ( *(_BYTE *)v12 != 1 || (v13 = *(_QWORD *)(v12 + 136), *(_BYTE *)v13 != 17) )
        BUG();
      v14 = *(_QWORD **)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) > 0x40u )
        v14 = (_QWORD *)*v14;
      v15 = v46;
      v16 = v10;
      v17 = (char *)&v46 + (_QWORD)*(v46 - 3);
      if ( !v10 )
      {
        if ( !v17[225] )
        {
          v31 = *((_QWORD *)v17 + 30);
          if ( !v31 )
LABEL_72:
            sub_426219();
          if ( !*(_BYTE *)(v31 + 56) )
          {
            v39 = *((_QWORD *)v17 + 30);
            sub_2216D60(v31, v5, v8, v6, (unsigned int)v10);
            v33 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v39 + 48LL);
            v15 = v46;
            if ( v33 != sub_CE72A0 )
            {
              v33(v39, 32u);
              v15 = v46;
            }
          }
          v17[225] = 1;
        }
        v17[224] = 48;
        v5 = (unsigned int)v14;
        *(__int64 *)((char *)&v48 + (_QWORD)*(v15 - 3)) = 8;
        *(_DWORD *)((char *)&v49 + (_QWORD)*(v46 - 3)) = *(_DWORD *)((_BYTE *)&v49 + (_QWORD)*(v46 - 3)) & 0xFFFFFFB5
                                                       | 8;
        sub_223E760(&v46, (unsigned int)v14);
        goto LABEL_17;
      }
      v18 = (unsigned __int8)v17[225];
      if ( (unsigned int)v10 > 3 )
      {
        if ( !(_BYTE)v18 )
        {
          v28 = *((_QWORD *)v17 + 30);
          if ( !v28 )
            goto LABEL_72;
          if ( !*(_BYTE *)(v28 + 56) )
          {
            v37 = *((_QWORD *)v17 + 30);
            sub_2216D60(v28, v18, v8, v6, (unsigned int)v10);
            v29 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v37 + 48LL);
            if ( v29 != sub_CE72A0 )
              v29(v37, 32u);
            v15 = v46;
          }
          v17[225] = 1;
        }
        v17[224] = 48;
        v5 = (unsigned __int8)v14;
        *(__int64 *)((char *)&v48 + (_QWORD)*(v15 - 3)) = 2;
        *(_DWORD *)((char *)&v49 + (_QWORD)*(v46 - 3)) = *(_DWORD *)((_BYTE *)&v49 + (_QWORD)*(v46 - 3)) & 0xFFFFFFB5
                                                       | 8;
        sub_223E730(&v46, (unsigned __int8)v14);
        v21 = (char *)&v46 + (_QWORD)*(v46 - 3);
        if ( !v21[225] )
        {
          v26 = *((_QWORD *)v21 + 30);
          if ( !v26 )
            goto LABEL_72;
          if ( !*(_BYTE *)(v26 + 56) )
          {
            v36 = *((_QWORD *)v21 + 30);
            sub_2216D60(v26, v5, v19, v6, v20);
            v27 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v36 + 48LL);
            if ( v27 != sub_CE72A0 )
            {
              v5 = 32;
              v27(v36, 32u);
            }
          }
          v21[225] = 1;
        }
        v21[224] = 32;
LABEL_17:
        if ( v9 == ++v10 )
          goto LABEL_24;
        goto LABEL_18;
      }
      if ( !(_BYTE)v18 )
      {
        v30 = *((_QWORD *)v17 + 30);
        if ( !v30 )
          goto LABEL_72;
        if ( *(_BYTE *)(v30 + 56) )
          goto LABEL_52;
        v38 = *((_QWORD *)v17 + 30);
        sub_2216D60(v30, v18, v8, v6, (unsigned int)v10);
        v16 = v10;
        v32 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v38 + 48LL);
        if ( v32 != sub_CE72A0 )
        {
          v32(v38, 32u);
          v15 = v46;
          v16 = v10;
LABEL_52:
          v17[225] = 1;
          goto LABEL_22;
        }
        v17[225] = 1;
        v15 = v46;
      }
LABEL_22:
      v17[224] = 48;
      v5 = (unsigned __int16)v14;
      v35 = v16;
      *(__int64 *)((char *)&v48 + (_QWORD)*(v15 - 3)) = 4;
      *(_DWORD *)((char *)&v49 + (_QWORD)*(v46 - 3)) = *(_DWORD *)((_BYTE *)&v49 + (_QWORD)*(v46 - 3)) & 0xFFFFFFB5 | 8;
      sub_223E760(&v46, (unsigned __int16)v14);
      if ( v35 != 2 )
        goto LABEL_17;
      v5 = (__int64)", 0x";
      ++v10;
      sub_223E0D0(&v46, ", 0x", 4);
      if ( v9 == v10 )
      {
LABEL_24:
        v3 = v34;
        break;
      }
LABEL_18:
      v7 = v40;
      v8 = (*(_BYTE *)(v40 - 16) & 2) != 0;
    }
  }
  n = 0;
  v41 = src;
  LOBYTE(src[0]) = 0;
  if ( v52 )
  {
    v22 = 0;
    if ( v52 <= v50 )
      sub_2241130(&v41, 0, 0, v51, v50 - v51);
    else
      sub_2241130(&v41, 0, 0, v51, v52 - v51);
  }
  else
  {
    v22 = v56;
    sub_2240AE0(&v41, v56);
  }
  v23 = (_BYTE *)*a2;
  v24 = n;
  if ( v41 == src )
  {
    if ( n )
    {
      if ( n == 1 )
      {
        *v23 = src[0];
      }
      else
      {
        v22 = src;
        memcpy(v23, src, n);
      }
      v24 = n;
      v23 = (_BYTE *)*a2;
    }
    a2[1] = v24;
    v23[v24] = 0;
    v23 = v41;
    goto LABEL_32;
  }
  v22 = (_QWORD *)src[0];
  if ( v23 == (_BYTE *)(a2 + 2) )
  {
    *a2 = v41;
    a2[1] = v24;
    a2[2] = v22;
    goto LABEL_68;
  }
  v25 = a2[2];
  *a2 = v41;
  a2[1] = v24;
  a2[2] = v22;
  if ( !v23 )
  {
LABEL_68:
    v41 = src;
    v23 = src;
    goto LABEL_32;
  }
  v41 = v23;
  src[0] = v25;
LABEL_32:
  n = 0;
  *v23 = 0;
  if ( v41 != src )
  {
    v22 = (_QWORD *)(src[0] + 1LL);
    j_j___libc_free_0(v41, src[0] + 1LL);
  }
  v44 = off_4A073F0;
  v58[0] = off_4A07440;
  v46 = off_4A07418;
  v47 = off_4A07080;
  if ( (_QWORD *)v56[0] != v57 )
  {
    v22 = (_QWORD *)(v57[0] + 1LL);
    j_j___libc_free_0(v56[0], v57[0] + 1LL);
  }
  v47 = off_4A07480;
  sub_2209150(v54, v22, v24);
  v44 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v44 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v46 = (__int64 (__fastcall **)())qword_4A07288;
  *(__int64 (__fastcall ***)())((char *)&v46 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
  v44 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v44 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v45 = 0;
  v58[0] = off_4A06798;
  sub_222E050(v58);
  return v3;
}
