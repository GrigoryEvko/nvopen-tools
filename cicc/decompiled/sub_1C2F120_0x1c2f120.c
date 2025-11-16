// Function: sub_1C2F120
// Address: 0x1c2f120
//
__int64 __fastcall sub_1C2F120(__int64 a1, _QWORD *a2)
{
  unsigned int v3; // r13d
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 (__fastcall **v11)(); // rax
  int v12; // r8d
  char *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r8
  char *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  _QWORD *v21; // rsi
  _BYTE *v22; // rdi
  size_t v23; // rdx
  __int64 v24; // rcx
  _BYTE *v25; // rdi
  __int64 (__fastcall *v26)(__int64, unsigned int); // rax
  __int64 v27; // rdi
  __int64 (__fastcall *v28)(__int64, unsigned int); // rax
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 (__fastcall *v31)(__int64, unsigned int); // rax
  __int64 (__fastcall *v32)(__int64, unsigned int); // r8
  char *v33; // [rsp+8h] [rbp-258h]
  _QWORD *v34; // [rsp+10h] [rbp-250h]
  unsigned __int8 v35; // [rsp+1Fh] [rbp-241h]
  int v36; // [rsp+68h] [rbp-1F8h]
  __int64 v37; // [rsp+68h] [rbp-1F8h]
  __int64 v38; // [rsp+68h] [rbp-1F8h]
  __int64 v39; // [rsp+68h] [rbp-1F8h]
  __int64 v40; // [rsp+78h] [rbp-1E8h] BYREF
  _QWORD *v41; // [rsp+80h] [rbp-1E0h] BYREF
  size_t n; // [rsp+88h] [rbp-1D8h]
  _QWORD src[2]; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 (__fastcall **v44)(); // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v45; // [rsp+A8h] [rbp-1B8h]
  __int64 (__fastcall **v46)(); // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 (__fastcall **v47)(); // [rsp+B8h] [rbp-1A8h] BYREF
  __int64 v48; // [rsp+C0h] [rbp-1A0h]
  __int64 v49; // [rsp+C8h] [rbp-198h]
  unsigned __int64 v50; // [rsp+D0h] [rbp-190h]
  __int64 v51; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v52; // [rsp+E0h] [rbp-180h]
  __int64 v53; // [rsp+E8h] [rbp-178h]
  char v54[8]; // [rsp+F0h] [rbp-170h] BYREF
  int v55; // [rsp+F8h] [rbp-168h]
  _QWORD v56[2]; // [rsp+100h] [rbp-160h] BYREF
  _QWORD v57[2]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v58[28]; // [rsp+120h] [rbp-140h] BYREF
  __int16 v59; // [rsp+200h] [rbp-60h]
  __int64 v60; // [rsp+208h] [rbp-58h]
  __int64 v61; // [rsp+210h] [rbp-50h]
  __int64 v62; // [rsp+218h] [rbp-48h]
  __int64 v63; // [rsp+220h] [rbp-40h]

  v40 = 0;
  v3 = sub_1C2E420(a1, "unified.entry.id", 0x10u, &v40);
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
  v49 = 0;
  v44 = off_4A073F0;
  v50 = 0;
  v51 = 0;
  v58[0] = off_4A07440;
  v52 = 0;
  v53 = 0;
  v46 = off_4A07418;
  v47 = off_4A07480;
  sub_220A990(v54);
  v55 = 24;
  LOBYTE(v57[0]) = 0;
  v47 = off_4A07080;
  v56[0] = v57;
  v56[1] = 0;
  sub_222DD70(v58, &v47);
  v5 = (unsigned __int64)"0x";
  sub_223E0D0(&v46, "0x", 2);
  v6 = v40;
  v7 = *(unsigned int *)(v40 + 8);
  if ( !(_DWORD)v7 )
    goto LABEL_19;
  v35 = v3;
  v8 = *(unsigned int *)(v40 + 8);
  v9 = 0;
  v34 = a2;
  while ( 1 )
  {
    v18 = sub_1C2E0F0(v6 + 8 * (v9 - v7));
    if ( *(_DWORD *)(v18 + 32) > 0x40u )
      v10 = **(_QWORD **)(v18 + 24);
    else
      v10 = *(_QWORD *)(v18 + 24);
    v11 = v46;
    v12 = v9;
    v13 = (char *)&v46 + (_QWORD)*(v46 - 3);
    if ( !v9 )
    {
      if ( !v13[225] )
      {
        v30 = *((_QWORD *)v13 + 30);
        if ( !v30 )
LABEL_65:
          sub_426219();
        if ( !*(_BYTE *)(v30 + 56) )
        {
          v39 = *((_QWORD *)v13 + 30);
          sub_2216D60(v30, v5, v19, v20, (unsigned int)v9);
          v32 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v39 + 48LL);
          v11 = v46;
          if ( v32 != sub_CE72A0 )
          {
            v32(v39, 32u);
            v11 = v46;
          }
        }
        v13[225] = 1;
      }
      v13[224] = 48;
      v5 = (unsigned int)v10;
      *(__int64 *)((char *)&v48 + (_QWORD)*(v11 - 3)) = 8;
      *(_DWORD *)((char *)&v49 + (_QWORD)*(v46 - 3)) = *(_DWORD *)((_BYTE *)&v49 + (_QWORD)*(v46 - 3)) & 0xFFFFFFB5 | 8;
      sub_223E760(&v46, (unsigned int)v10);
      goto LABEL_11;
    }
    v14 = (unsigned __int8)v13[225];
    if ( (unsigned int)v9 > 3 )
    {
      if ( !(_BYTE)v14 )
      {
        v27 = *((_QWORD *)v13 + 30);
        if ( !v27 )
          goto LABEL_65;
        if ( !*(_BYTE *)(v27 + 56) )
        {
          v37 = *((_QWORD *)v13 + 30);
          sub_2216D60(v27, v14, v19, v20, (unsigned int)v9);
          v28 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v37 + 48LL);
          if ( v28 != sub_CE72A0 )
            v28(v37, 32u);
          v11 = v46;
        }
        v13[225] = 1;
      }
      v13[224] = 48;
      v5 = (unsigned int)v10;
      *(__int64 *)((char *)&v48 + (_QWORD)*(v11 - 3)) = 2;
      *(_DWORD *)((char *)&v49 + (_QWORD)*(v46 - 3)) = *(_DWORD *)((_BYTE *)&v49 + (_QWORD)*(v46 - 3)) & 0xFFFFFFB5 | 8;
      sub_223E730(&v46, (unsigned int)v10);
      v17 = (char *)&v46 + (_QWORD)*(v46 - 3);
      if ( !v17[225] )
      {
        v25 = (_BYTE *)*((_QWORD *)v17 + 30);
        if ( !v25 )
          goto LABEL_65;
        if ( !v25[56] )
        {
          v33 = (char *)&v46 + (_QWORD)*(v46 - 3);
          sub_2216D60(v25, (unsigned int)v10, v17, v15, v16);
          v17 = v33;
          v26 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v25 + 48LL);
          if ( v26 != sub_CE72A0 )
          {
            v5 = 32;
            v26((__int64)v25, 32u);
            v17 = v33;
          }
        }
        v17[225] = 1;
      }
      v17[224] = 32;
      goto LABEL_11;
    }
    if ( !(_BYTE)v14 )
    {
      v29 = *((_QWORD *)v13 + 30);
      if ( !v29 )
        goto LABEL_65;
      if ( *(_BYTE *)(v29 + 56) )
        goto LABEL_46;
      v38 = *((_QWORD *)v13 + 30);
      sub_2216D60(v29, v14, v19, v20, (unsigned int)v9);
      v12 = v9;
      v31 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v38 + 48LL);
      if ( v31 != sub_CE72A0 )
      {
        v31(v38, 32u);
        v11 = v46;
        v12 = v9;
LABEL_46:
        v13[225] = 1;
        goto LABEL_16;
      }
      v13[225] = 1;
      v11 = v46;
    }
LABEL_16:
    v13[224] = 48;
    v5 = (unsigned int)v10;
    v36 = v12;
    *(__int64 *)((char *)&v48 + (_QWORD)*(v11 - 3)) = 4;
    *(_DWORD *)((char *)&v49 + (_QWORD)*(v46 - 3)) = *(_DWORD *)((_BYTE *)&v49 + (_QWORD)*(v46 - 3)) & 0xFFFFFFB5 | 8;
    sub_223E760(&v46, (unsigned int)v10);
    if ( v36 == 2 )
      break;
LABEL_11:
    if ( v8 == ++v9 )
      goto LABEL_18;
LABEL_12:
    v6 = v40;
    v7 = *(unsigned int *)(v40 + 8);
  }
  v5 = (unsigned __int64)", 0x";
  ++v9;
  sub_223E0D0(&v46, ", 0x", 4);
  if ( v8 != v9 )
    goto LABEL_12;
LABEL_18:
  v3 = v35;
  a2 = v34;
LABEL_19:
  n = 0;
  v41 = src;
  LOBYTE(src[0]) = 0;
  if ( v52 )
  {
    v21 = 0;
    if ( v52 <= v50 )
      sub_2241130(&v41, 0, 0, v51, v50 - v51);
    else
      sub_2241130(&v41, 0, 0, v51, v52 - v51);
  }
  else
  {
    v21 = v56;
    sub_2240AE0(&v41, v56);
  }
  v22 = (_BYTE *)*a2;
  v23 = n;
  if ( v41 == src )
  {
    if ( n )
    {
      if ( n == 1 )
      {
        *v22 = src[0];
      }
      else
      {
        v21 = src;
        memcpy(v22, src, n);
      }
      v23 = n;
      v22 = (_BYTE *)*a2;
    }
    a2[1] = v23;
    v22[v23] = 0;
    v22 = v41;
  }
  else
  {
    v21 = (_QWORD *)src[0];
    if ( v22 == (_BYTE *)(a2 + 2) )
    {
      *a2 = v41;
      a2[1] = v23;
      a2[2] = v21;
    }
    else
    {
      v24 = a2[2];
      *a2 = v41;
      a2[1] = v23;
      a2[2] = v21;
      if ( v22 )
      {
        v41 = v22;
        src[0] = v24;
        goto LABEL_26;
      }
    }
    v41 = src;
    v22 = src;
  }
LABEL_26:
  n = 0;
  *v22 = 0;
  if ( v41 != src )
  {
    v21 = (_QWORD *)(src[0] + 1LL);
    j_j___libc_free_0(v41, src[0] + 1LL);
  }
  v44 = off_4A073F0;
  v58[0] = off_4A07440;
  v46 = off_4A07418;
  v47 = off_4A07080;
  if ( (_QWORD *)v56[0] != v57 )
  {
    v21 = (_QWORD *)(v57[0] + 1LL);
    j_j___libc_free_0(v56[0], v57[0] + 1LL);
  }
  v47 = off_4A07480;
  sub_2209150(v54, v21, v23);
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
