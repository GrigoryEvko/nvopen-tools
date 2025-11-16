// Function: sub_214C1B0
// Address: 0x214c1b0
//
__int64 (*__fastcall sub_214C1B0(_QWORD *a1, __int64 a2))(void)
{
  int v3; // r8d
  __int64 v4; // rdi
  __int64 (*v5)(void); // rax
  __int64 v6; // r12
  char *v7; // rsi
  char *v8; // rax
  char *v9; // r9
  char *v10; // r8
  size_t v11; // rax
  _BYTE *v12; // r10
  size_t v13; // r11
  __int64 v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // r12
  bool v17; // zf
  void (*v18)(); // rax
  __int64 (*result)(void); // rax
  void (*v20)(); // r13
  __int64 v21; // rax
  char *v22; // rdi
  __int64 v23; // rax
  _BYTE *dest; // [rsp+10h] [rbp-250h]
  void *desta; // [rsp+10h] [rbp-250h]
  char *v26; // [rsp+18h] [rbp-248h]
  char *v27; // [rsp+18h] [rbp-248h]
  unsigned int sa; // [rsp+50h] [rbp-210h]
  char *s; // [rsp+50h] [rbp-210h]
  char *sc; // [rsp+50h] [rbp-210h]
  char *sd; // [rsp+50h] [rbp-210h]
  char *sb; // [rsp+50h] [rbp-210h]
  int v33; // [rsp+58h] [rbp-208h]
  void (*v34)(); // [rsp+60h] [rbp-200h]
  __int64 v35; // [rsp+68h] [rbp-1F8h]
  size_t v36; // [rsp+78h] [rbp-1E8h] BYREF
  const char *v37; // [rsp+80h] [rbp-1E0h] BYREF
  const char **v38; // [rsp+88h] [rbp-1D8h]
  _QWORD v39[2]; // [rsp+90h] [rbp-1D0h] BYREF
  const char *v40; // [rsp+A0h] [rbp-1C0h] BYREF
  _BYTE *v41; // [rsp+A8h] [rbp-1B8h]
  __int64 (__fastcall **v42)(); // [rsp+B0h] [rbp-1B0h] BYREF
  _QWORD v43[3]; // [rsp+B8h] [rbp-1A8h] BYREF
  unsigned __int64 v44; // [rsp+D0h] [rbp-190h]
  __int64 v45; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v46; // [rsp+E0h] [rbp-180h]
  __int64 v47; // [rsp+E8h] [rbp-178h]
  char v48[8]; // [rsp+F0h] [rbp-170h] BYREF
  int v49; // [rsp+F8h] [rbp-168h]
  _QWORD v50[2]; // [rsp+100h] [rbp-160h] BYREF
  _QWORD v51[2]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v52[28]; // [rsp+120h] [rbp-140h] BYREF
  __int16 v53; // [rsp+200h] [rbp-60h]
  __int64 v54; // [rsp+208h] [rbp-58h]
  __int64 v55; // [rsp+210h] [rbp-50h]
  __int64 v56; // [rsp+218h] [rbp-48h]
  __int64 v57; // [rsp+220h] [rbp-40h]

  v3 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
  v35 = a1[32];
  if ( v3 < 0 )
  {
    v20 = *(void (**)())(*(_QWORD *)a1[32] + 104LL);
    (*(void (__fastcall **)(const char **, _QWORD *, _QWORD))(*a1 + 392LL))(&v40, a1, (unsigned int)v3);
    v37 = "implicit-def: ";
    v38 = &v40;
    LOWORD(v39[0]) = 1027;
    if ( v20 != nullsub_580 )
      ((void (__fastcall *)(__int64, const char **, __int64))v20)(v35, &v37, 1);
    if ( v40 != (const char *)&v42 )
      j_j___libc_free_0(v40, (char *)v42 + 1);
    goto LABEL_21;
  }
  v4 = a1[105];
  v34 = *(void (**)())(*(_QWORD *)v35 + 104LL);
  v5 = *(__int64 (**)(void))(*(_QWORD *)v4 + 112LL);
  if ( (char *)v5 == (char *)sub_214AB90 )
  {
    v6 = v4 + 320;
  }
  else
  {
    v33 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    v21 = v5();
    v3 = v33;
    v6 = v21;
  }
  sa = v3;
  sub_222DF20(v52);
  v53 = 0;
  v52[27] = 0;
  v52[0] = off_4A06798;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v40 = (const char *)qword_4A072D8;
  v57 = 0;
  *(const char **)((char *)&v40 + qword_4A072D8[-3]) = (const char *)&unk_4A07300;
  v41 = 0;
  sub_222DD70((char *)&v40 + *((_QWORD *)v40 - 3), 0);
  v42 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v43[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((char *)&v43[-1] + (_QWORD)*(v42 - 3), 0);
  v40 = (const char *)qword_4A07328;
  *(const char **)((char *)&v40 + qword_4A07328[-3]) = (const char *)&unk_4A07378;
  v43[1] = 0;
  v43[2] = 0;
  v40 = (const char *)off_4A073F0;
  v44 = 0;
  v45 = 0;
  v52[0] = off_4A07440;
  v46 = 0;
  v47 = 0;
  v42 = off_4A07418;
  v43[0] = off_4A07480;
  sub_220A990(v48);
  v49 = 24;
  LOBYTE(v51[0]) = 0;
  v43[0] = off_4A07080;
  v50[0] = v51;
  v50[1] = 0;
  sub_222DD70(v52, v43);
  sub_223E0D0(&v42, "reg", 3);
  sub_223E760(&v42, sa);
  v38 = 0;
  v37 = (const char *)v39;
  LOBYTE(v39[0]) = 0;
  if ( v46 )
  {
    v7 = 0;
    if ( v46 <= v44 )
      sub_2241130(&v37, 0, 0, v45, v44 - v45);
    else
      sub_2241130(&v37, 0, 0, v45, v46 - v45);
  }
  else
  {
    v7 = (char *)v50;
    sub_2240AE0(&v37, v50);
  }
  s = (char *)v37;
  v8 = (char *)sub_22077B0(32);
  v10 = v8;
  if ( v8 )
  {
    v26 = v8;
    *(_QWORD *)v8 = v8 + 16;
    dest = v8 + 16;
    if ( !s )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v11 = strlen(s);
    v9 = s;
    v10 = v26;
    v36 = v11;
    v12 = dest;
    v13 = v11;
    if ( v11 > 0xF )
    {
      v22 = v26;
      v27 = s;
      sc = v10;
      desta = (void *)v11;
      v23 = sub_22409D0(v22, &v36, 0);
      v10 = sc;
      v9 = v27;
      v12 = (_BYTE *)v23;
      v13 = (size_t)desta;
      *(_QWORD *)sc = v23;
      *((_QWORD *)sc + 2) = v36;
    }
    else
    {
      if ( v11 == 1 )
      {
        v26[16] = *s;
LABEL_12:
        *((_QWORD *)v10 + 1) = v11;
        v12[v11] = 0;
        goto LABEL_13;
      }
      if ( !v11 )
        goto LABEL_12;
    }
    v7 = v9;
    sd = v10;
    memcpy(v12, v9, v13);
    v10 = sd;
    v11 = v36;
    v12 = *(_BYTE **)sd;
    goto LABEL_12;
  }
LABEL_13:
  v14 = *(unsigned int *)(v6 + 304);
  if ( (unsigned int)v14 >= *(_DWORD *)(v6 + 308) )
  {
    v7 = (char *)(v6 + 312);
    sb = v10;
    sub_16CD150(v6 + 296, (const void *)(v6 + 312), 0, 8, (int)v10, (int)v9);
    v14 = *(unsigned int *)(v6 + 304);
    v10 = sb;
  }
  v15 = *(_QWORD *)(v6 + 296);
  *(_QWORD *)(v15 + 8 * v14) = v10;
  ++*(_DWORD *)(v6 + 304);
  v16 = *(_BYTE **)v10;
  if ( v37 != (const char *)v39 )
  {
    v7 = (char *)(v39[0] + 1LL);
    j_j___libc_free_0(v37, v39[0] + 1LL);
  }
  v40 = (const char *)off_4A073F0;
  v52[0] = off_4A07440;
  v42 = off_4A07418;
  v43[0] = off_4A07080;
  if ( (_QWORD *)v50[0] != v51 )
  {
    v7 = (char *)(v51[0] + 1LL);
    j_j___libc_free_0(v50[0], v51[0] + 1LL);
  }
  v43[0] = off_4A07480;
  sub_2209150(v48, v7, v15);
  v40 = (const char *)qword_4A07328;
  *(const char **)((char *)&v40 + qword_4A07328[-3]) = (const char *)&unk_4A07378;
  v42 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v43[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v40 = (const char *)qword_4A072D8;
  *(const char **)((char *)&v40 + qword_4A072D8[-3]) = (const char *)&unk_4A07300;
  v41 = 0;
  v52[0] = off_4A06798;
  sub_222E050(v52);
  v17 = *v16 == 0;
  v40 = "implicit-def: ";
  if ( v17 )
  {
    LOWORD(v42) = 259;
    v18 = v34;
    if ( v34 == nullsub_580 )
      goto LABEL_21;
  }
  else
  {
    v18 = v34;
    v41 = v16;
    LOWORD(v42) = 771;
    if ( v34 == nullsub_580 )
      goto LABEL_21;
  }
  ((void (__fastcall *)(__int64, const char **, __int64))v18)(v35, &v40, 1);
LABEL_21:
  result = *(__int64 (**)(void))(*(_QWORD *)a1[32] + 144LL);
  if ( (char *)result != (char *)nullsub_581 )
    return (__int64 (*)(void))result();
  return result;
}
