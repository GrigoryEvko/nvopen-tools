// Function: sub_1F3DC60
// Address: 0x1f3dc60
//
__int64 __fastcall sub_1F3DC60(char a1, unsigned int a2, __int64 a3, char *a4, unsigned __int64 a5, int a6)
{
  unsigned int v6; // r13d
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  char *v11; // r12
  unsigned __int64 v12; // rbx
  char v13; // cl
  _QWORD *v14; // r10
  int v15; // eax
  int v16; // eax
  bool v17; // zf
  void *v18; // rax
  char v19; // [rsp+Fh] [rbp-101h]
  char v20; // [rsp+10h] [rbp-100h]
  _QWORD *v21; // [rsp+10h] [rbp-100h]
  _BYTE *v22; // [rsp+28h] [rbp-E8h]
  char *v23; // [rsp+30h] [rbp-E0h] BYREF
  unsigned __int64 v24; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v25; // [rsp+47h] [rbp-C9h] BYREF
  unsigned __int64 v26; // [rsp+48h] [rbp-C8h] BYREF
  void *s2; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-B8h]
  __int64 v29; // [rsp+60h] [rbp-B0h] BYREF
  void *v30; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v31; // [rsp+78h] [rbp-98h]
  _QWORD v32[2]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v33; // [rsp+90h] [rbp-80h] BYREF
  __int64 v34; // [rsp+98h] [rbp-78h]
  _BYTE v35[112]; // [rsp+A0h] [rbp-70h] BYREF

  v6 = -1;
  v23 = a4;
  v24 = a5;
  if ( !a5 )
    return v6;
  v33 = v35;
  v34 = 0x400000000LL;
  sub_16D2880(&v23, (__int64)&v33, 44, -1, 1, a6);
  if ( (_DWORD)v34 == 1 )
  {
    v17 = (unsigned __int8)sub_1F3CD70(v23, v24, (signed __int64 *)&v30, (unsigned __int8 *)&s2) == 0;
    v18 = (void *)v24;
    if ( !v17 )
    {
      if ( (unsigned __int64)v30 <= v24 )
        v18 = v30;
      v24 = (unsigned __int64)v18;
    }
    if ( v18 == (void *)3 )
    {
      if ( *(_WORD *)v23 == 27745 )
      {
        v6 = 1;
        if ( v23[2] == 108 )
          goto LABEL_24;
      }
    }
    else if ( v18 == (void *)4 )
    {
      v6 = 0;
      if ( *(_DWORD *)v23 == 1701736302 )
        goto LABEL_24;
    }
    else if ( v18 == (void *)7 && *(_DWORD *)v23 == 1634100580 && *((_WORD *)v23 + 2) == 27765 )
    {
      v6 = -1;
      if ( v23[6] == 116 )
        goto LABEL_24;
    }
  }
  sub_1F3DAC0((__int64)&s2, a1, a2, a3);
  v30 = v32;
  sub_1F3D130((__int64 *)&v30, s2, (__int64)s2 + v28);
  sub_2240CE0(&v30, v31 - 1, 1);
  v9 = 16LL * (unsigned int)v34;
  v22 = &v33[v9];
  if ( &v33[v9] == v33 )
  {
    v14 = v30;
LABEL_13:
    v6 = -1;
    goto LABEL_20;
  }
  v10 = (unsigned __int64)v33;
  while ( 1 )
  {
    v11 = *(char **)v10;
    v12 = *(_QWORD *)(v10 + 8);
    if ( (unsigned __int8)sub_1F3CD70(*(char **)v10, v12, (signed __int64 *)&v26, &v25) && v12 > v26 )
      v12 = v26;
    v13 = *v11;
    if ( *v11 != 33 )
      break;
    if ( v12 )
    {
      --v12;
      ++v11;
      if ( v28 == v12 )
        goto LABEL_16;
LABEL_11:
      v14 = v30;
      if ( v31 != v12 )
        goto LABEL_12;
      if ( !v12 )
        goto LABEL_19;
LABEL_27:
      v19 = v13;
      v21 = v14;
      v16 = memcmp(v11, v14, v12);
      v14 = v21;
      v13 = v19;
      if ( !v16 )
        goto LABEL_19;
      goto LABEL_12;
    }
    if ( !v28 )
      goto LABEL_18;
    v14 = v30;
    if ( !v31 )
      goto LABEL_19;
LABEL_12:
    v10 += 16LL;
    if ( v22 == (_BYTE *)v10 )
      goto LABEL_13;
  }
  if ( v28 != v12 )
    goto LABEL_11;
LABEL_16:
  if ( v12 )
  {
    v20 = v13;
    v15 = memcmp(v11, s2, v12);
    v13 = v20;
    if ( v15 )
    {
      v14 = v30;
      if ( v31 != v12 )
        goto LABEL_12;
      goto LABEL_27;
    }
  }
LABEL_18:
  v14 = v30;
LABEL_19:
  v6 = v13 != 33;
LABEL_20:
  if ( v14 != v32 )
    j_j___libc_free_0(v14, v32[0] + 1LL);
  if ( s2 != &v29 )
    j_j___libc_free_0(s2, v29 + 1);
LABEL_24:
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  return v6;
}
