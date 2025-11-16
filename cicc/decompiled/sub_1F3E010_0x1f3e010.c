// Function: sub_1F3E010
// Address: 0x1f3e010
//
__int64 __fastcall sub_1F3E010(char a1, unsigned int a2, __int64 a3, char *a4, unsigned __int64 a5, int a6)
{
  __int64 v7; // rbx
  unsigned __int64 v8; // r15
  char *v9; // r14
  size_t v10; // rbx
  size_t v11; // rdx
  _QWORD *v12; // r9
  unsigned int v13; // r13d
  int v14; // eax
  int v16; // eax
  void *v17; // rax
  _QWORD *v18; // [rsp+0h] [rbp-100h]
  _BYTE *v19; // [rsp+18h] [rbp-E8h]
  char *v20; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v22; // [rsp+37h] [rbp-C9h] BYREF
  size_t n; // [rsp+38h] [rbp-C8h] BYREF
  void *s2; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+48h] [rbp-B8h]
  __int64 v26; // [rsp+50h] [rbp-B0h] BYREF
  void *v27; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+68h] [rbp-98h]
  _QWORD v29[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE *v30; // [rsp+80h] [rbp-80h] BYREF
  __int64 v31; // [rsp+88h] [rbp-78h]
  _BYTE v32[112]; // [rsp+90h] [rbp-70h] BYREF

  v20 = a4;
  v21 = a5;
  if ( !a5 )
    return (unsigned int)-1;
  v30 = v32;
  v31 = 0x400000000LL;
  sub_16D2880(&v20, (__int64)&v30, 44, -1, 1, a6);
  if ( (_DWORD)v31 != 1 )
    goto LABEL_3;
  v13 = -1;
  if ( !(unsigned __int8)sub_1F3CD70(v20, v21, (signed __int64 *)&v27, (unsigned __int8 *)&s2) )
    goto LABEL_19;
  v17 = (void *)v21;
  if ( (unsigned __int64)v27 <= v21 )
    v17 = v27;
  v21 = (unsigned __int64)v17;
  if ( v17 == (void *)3 )
  {
    if ( *(_WORD *)v20 == 27745 && v20[2] == 108 )
      goto LABEL_36;
  }
  else if ( v17 == (void *)7 && *(_DWORD *)v20 == 1634100580 && *((_WORD *)v20 + 2) == 27765 && v20[6] == 116 )
  {
LABEL_36:
    v13 = (unsigned __int8)s2;
    goto LABEL_19;
  }
LABEL_3:
  sub_1F3DAC0((__int64)&s2, a1, a2, a3);
  v27 = v29;
  sub_1F3D130((__int64 *)&v27, s2, (__int64)s2 + v25);
  sub_2240CE0(&v27, v28 - 1, 1);
  v7 = 16LL * (unsigned int)v31;
  v8 = (unsigned __int64)v30;
  v19 = &v30[v7];
  if ( &v30[v7] == v30 )
  {
LABEL_10:
    v12 = v27;
    v13 = -1;
    goto LABEL_15;
  }
  while ( 1 )
  {
    v9 = *(char **)v8;
    v10 = *(_QWORD *)(v8 + 8);
    if ( !(unsigned __int8)sub_1F3CD70(*(char **)v8, v10, (signed __int64 *)&n, &v22) )
      goto LABEL_9;
    if ( n <= v10 )
      v10 = n;
    v11 = v10;
    if ( v10 == v25 )
    {
      if ( !v10 )
        break;
      v14 = memcmp(v9, s2, v10);
      v11 = v10;
      if ( !v14 )
        break;
    }
    if ( v11 == v28 )
    {
      v12 = v27;
      if ( !v11 )
        goto LABEL_14;
      v18 = v27;
      v16 = memcmp(v9, v27, v11);
      v12 = v18;
      if ( !v16 )
        goto LABEL_14;
    }
LABEL_9:
    v8 += 16LL;
    if ( v19 == (_BYTE *)v8 )
      goto LABEL_10;
  }
  v12 = v27;
LABEL_14:
  v13 = v22;
LABEL_15:
  if ( v12 != v29 )
    j_j___libc_free_0(v12, v29[0] + 1LL);
  if ( s2 != &v26 )
    j_j___libc_free_0(s2, v26 + 1);
LABEL_19:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v13;
}
