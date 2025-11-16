// Function: sub_16B01B0
// Address: 0x16b01b0
//
__int64 __fastcall sub_16B01B0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        char *a5,
        __int64 a6,
        unsigned __int8 a7)
{
  unsigned int v7; // r10d
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 result; // rax
  size_t v13; // rdx
  char *v14; // rax
  char *v15; // r12
  __int64 v16; // r10
  char *v17; // r13
  __int64 v18; // r9
  unsigned __int64 v19; // r12
  size_t v20; // rdx
  _BYTE *v21; // rax
  char *v22; // r8
  char *v23; // [rsp+8h] [rbp-38h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  v7 = a2;
  v10 = a1;
  v11 = a6;
  if ( (*(_BYTE *)(a1 + 13) & 2) == 0 || !a6 )
    return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, __int64))(*(_QWORD *)v10 + 80LL))(
             v10,
             v7,
             a3,
             a4,
             a5,
             v11);
  v13 = 0x7FFFFFFFFFFFFFFFLL;
  if ( a6 >= 0 )
    v13 = a6;
  v23 = a5;
  v14 = (char *)memchr(a5, 44, v13);
  a5 = v23;
  v7 = a2;
  if ( !v14 )
    return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, __int64))(*(_QWORD *)v10 + 80LL))(
             v10,
             v7,
             a3,
             a4,
             a5,
             v11);
  v15 = (char *)(v14 - v23);
  if ( v14 - v23 == -1 )
    return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, __int64))(*(_QWORD *)v10 + 80LL))(
             v10,
             v7,
             a3,
             a4,
             a5,
             v11);
  v16 = a1;
  v17 = v23;
  while ( 1 )
  {
    v18 = (__int64)v15;
    if ( v11 <= (unsigned __int64)v15 )
      v18 = v11;
    v24 = v16;
    result = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, __int64, _QWORD))(*(_QWORD *)v16 + 80LL))(
               v16,
               a2,
               a3,
               a4,
               v17,
               v18,
               a7);
    if ( (_BYTE)result )
      return result;
    v19 = (unsigned __int64)(v15 + 1);
    v16 = v24;
    if ( v19 > v11 )
    {
      v22 = v17;
      v10 = v24;
      v7 = a2;
      a5 = &v22[v11];
      v11 = 0;
      return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, __int64))(*(_QWORD *)v10 + 80LL))(
               v10,
               v7,
               a3,
               a4,
               a5,
               v11);
    }
    v11 -= v19;
    v17 += v19;
    if ( v11 == -1 )
    {
      v20 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else
    {
      if ( !v11 )
        goto LABEL_17;
      v20 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v11 >= 0 )
        v20 = v11;
    }
    v21 = memchr(v17, 44, v20);
    v16 = v24;
    if ( v21 )
    {
      v15 = (char *)(v21 - v17);
      if ( v21 - v17 != -1 )
        continue;
    }
LABEL_17:
    a5 = v17;
    v10 = v16;
    v7 = a2;
    return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, char *, __int64))(*(_QWORD *)v10 + 80LL))(
             v10,
             v7,
             a3,
             a4,
             a5,
             v11);
  }
}
