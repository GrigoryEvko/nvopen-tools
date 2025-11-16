// Function: sub_2FE4ED0
// Address: 0x2fe4ed0
//
__int64 __fastcall sub_2FE4ED0(unsigned __int8 a1, unsigned int a2, __int64 a3, char *a4, size_t a5, __int64 a6)
{
  unsigned int v6; // r13d
  unsigned __int64 v9; // rbx
  char *v10; // r12
  size_t v11; // r13
  char v12; // r15
  bool v13; // zf
  void *v14; // rax
  _QWORD *v15; // r14
  _BYTE *v16; // [rsp+18h] [rbp-F8h]
  char *v17; // [rsp+30h] [rbp-E0h] BYREF
  size_t v18; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v19; // [rsp+47h] [rbp-C9h] BYREF
  unsigned __int64 v20; // [rsp+48h] [rbp-C8h] BYREF
  void *s2; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+58h] [rbp-B8h]
  __int64 v23; // [rsp+60h] [rbp-B0h] BYREF
  void *v24; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+78h] [rbp-98h]
  _QWORD v26[2]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v27; // [rsp+90h] [rbp-80h] BYREF
  __int64 v28; // [rsp+98h] [rbp-78h]
  _BYTE v29[112]; // [rsp+A0h] [rbp-70h] BYREF

  v6 = -1;
  v17 = a4;
  v18 = a5;
  if ( !a5 )
    return v6;
  v27 = v29;
  v28 = 0x400000000LL;
  sub_C93960(&v17, (__int64)&v27, 44, -1, 1, a6);
  if ( (_DWORD)v28 != 1 )
  {
LABEL_4:
    sub_2FE4690((__int64)&s2, a1, a2, a3);
    v24 = v26;
    sub_2FE3DD0((__int64 *)&v24, s2, (__int64)s2 + v22);
    sub_2240CE0((__int64 *)&v24, v25 - 1, 1);
    v9 = (unsigned __int64)v27;
    v16 = &v27[16 * (unsigned int)v28];
    if ( v16 == v27 )
    {
      v15 = v24;
LABEL_41:
      v6 = -1;
LABEL_19:
      if ( v15 != v26 )
        j_j___libc_free_0((unsigned __int64)v15);
      if ( s2 != &v23 )
        j_j___libc_free_0((unsigned __int64)s2);
      goto LABEL_23;
    }
    while ( 1 )
    {
      v10 = *(char **)v9;
      v11 = *(_QWORD *)(v9 + 8);
      if ( (unsigned __int8)sub_2FE4260(*(_BYTE **)v9, v11, (signed __int64 *)&v20, &v19) && v11 > v20 )
        v11 = v20;
      v12 = *v10;
      if ( *v10 == 33 )
      {
        if ( !v11 )
        {
          v15 = v24;
          if ( !v22 || !v25 )
          {
LABEL_18:
            v6 = v12 != 33;
            goto LABEL_19;
          }
          goto LABEL_10;
        }
        --v11;
        ++v10;
      }
      v15 = v24;
      if ( v22 != v11 )
      {
        if ( v25 != v11 )
          goto LABEL_10;
        if ( !v11 )
          goto LABEL_18;
LABEL_9:
        if ( !memcmp(v10, v15, v11) )
          goto LABEL_18;
        goto LABEL_10;
      }
      if ( !v11 || !memcmp(v10, s2, v11) )
        goto LABEL_18;
      if ( v25 == v11 )
        goto LABEL_9;
LABEL_10:
      v9 += 16LL;
      if ( v16 == (_BYTE *)v9 )
        goto LABEL_41;
    }
  }
  v13 = (unsigned __int8)sub_2FE4260(v17, v18, (signed __int64 *)&v24, (unsigned __int8 *)&s2) == 0;
  v14 = (void *)v18;
  if ( !v13 )
  {
    if ( (unsigned __int64)v24 <= v18 )
      v14 = v24;
    v18 = (size_t)v14;
  }
  if ( v14 == (void *)3 )
  {
    if ( *(_WORD *)v17 != 27745 )
      goto LABEL_4;
    v6 = 1;
    if ( v17[2] != 108 )
      goto LABEL_4;
  }
  else if ( v14 == (void *)4 )
  {
    v6 = 0;
    if ( *(_DWORD *)v17 != 1701736302 )
      goto LABEL_4;
  }
  else
  {
    if ( v14 != (void *)7 )
      goto LABEL_4;
    if ( *(_DWORD *)v17 != 1634100580 )
      goto LABEL_4;
    if ( *((_WORD *)v17 + 2) != 27765 )
      goto LABEL_4;
    v6 = -1;
    if ( v17[6] != 116 )
      goto LABEL_4;
  }
LABEL_23:
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v6;
}
