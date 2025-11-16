// Function: sub_2FE4940
// Address: 0x2fe4940
//
__int64 __fastcall sub_2FE4940(unsigned __int8 a1, unsigned int a2, __int64 a3, char *a4, size_t a5, __int64 a6)
{
  __int64 v7; // rbx
  unsigned __int64 v8; // r14
  int v9; // eax
  size_t v10; // rdx
  const void *v11; // r15
  size_t v12; // rbx
  _QWORD *v13; // rbx
  unsigned int v14; // r13d
  void *v16; // rax
  size_t v17; // [rsp+10h] [rbp-F0h]
  _BYTE *v18; // [rsp+18h] [rbp-E8h]
  char *v19; // [rsp+20h] [rbp-E0h] BYREF
  size_t v20; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v21; // [rsp+37h] [rbp-C9h] BYREF
  size_t n; // [rsp+38h] [rbp-C8h] BYREF
  void *s2; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+48h] [rbp-B8h]
  __int64 v25; // [rsp+50h] [rbp-B0h] BYREF
  void *v26; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-98h]
  _QWORD v28[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE *v29; // [rsp+80h] [rbp-80h] BYREF
  __int64 v30; // [rsp+88h] [rbp-78h]
  _BYTE v31[112]; // [rsp+90h] [rbp-70h] BYREF

  v19 = a4;
  v20 = a5;
  if ( !a5 )
    return (unsigned int)-1;
  v29 = v31;
  v30 = 0x400000000LL;
  sub_C93960(&v19, (__int64)&v29, 44, -1, 1, a6);
  if ( (_DWORD)v30 != 1 )
  {
LABEL_3:
    sub_2FE4690((__int64)&s2, a1, a2, a3);
    v26 = v28;
    sub_2FE3DD0((__int64 *)&v26, s2, (__int64)s2 + v24);
    sub_2240CE0((__int64 *)&v26, v27 - 1, 1);
    v7 = 16LL * (unsigned int)v30;
    v8 = (unsigned __int64)v29;
    v18 = &v29[v7];
    if ( &v29[v7] == v29 )
    {
LABEL_32:
      v13 = v26;
      v14 = -1;
      if ( v26 != v28 )
LABEL_17:
        j_j___libc_free_0((unsigned __int64)v13);
LABEL_18:
      if ( s2 != &v25 )
        j_j___libc_free_0((unsigned __int64)s2);
      goto LABEL_20;
    }
    while ( 1 )
    {
      v11 = *(const void **)v8;
      v12 = *(_QWORD *)(v8 + 8);
      if ( !(unsigned __int8)sub_2FE4260(*(_BYTE **)v8, v12, (signed __int64 *)&n, &v21) )
        goto LABEL_9;
      if ( n <= v12 )
        v12 = n;
      v10 = v12;
      v13 = v26;
      if ( v24 != v10 )
        break;
      if ( !v10 || (v17 = v10, v9 = memcmp(v11, s2, v10), v10 = v17, !v9) )
      {
LABEL_16:
        v14 = v21;
        if ( v13 != v28 )
          goto LABEL_17;
        goto LABEL_18;
      }
      if ( v17 == v27 )
        goto LABEL_8;
LABEL_9:
      v8 += 16LL;
      if ( v18 == (_BYTE *)v8 )
        goto LABEL_32;
    }
    if ( v27 != v10 )
      goto LABEL_9;
    if ( !v10 )
      goto LABEL_16;
LABEL_8:
    if ( !memcmp(v11, v13, v10) )
      goto LABEL_16;
    goto LABEL_9;
  }
  v14 = -1;
  if ( (unsigned __int8)sub_2FE4260(v19, v20, (signed __int64 *)&v26, (unsigned __int8 *)&s2) )
  {
    v16 = (void *)v20;
    if ( (unsigned __int64)v26 <= v20 )
      v16 = v26;
    v20 = (size_t)v16;
    if ( v16 == (void *)3 )
    {
      if ( *(_WORD *)v19 != 27745 || v19[2] != 108 )
        goto LABEL_3;
    }
    else if ( v16 != (void *)7 || *(_DWORD *)v19 != 1634100580 || *((_WORD *)v19 + 2) != 27765 || v19[6] != 116 )
    {
      goto LABEL_3;
    }
    v14 = (unsigned __int8)s2;
  }
LABEL_20:
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
  return v14;
}
