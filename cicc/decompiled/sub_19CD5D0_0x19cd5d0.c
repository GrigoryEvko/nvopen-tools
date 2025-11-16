// Function: sub_19CD5D0
// Address: 0x19cd5d0
//
__int64 __fastcall sub_19CD5D0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r14d
  void **v12; // rax
  void **v13; // rbx
  void **v14; // rdx
  char v15[8]; // [rsp+70h] [rbp-A0h] BYREF
  void **v16; // [rsp+78h] [rbp-98h]
  void **v17; // [rsp+80h] [rbp-90h]
  int v18; // [rsp+88h] [rbp-88h]
  int v19; // [rsp+8Ch] [rbp-84h]
  __int64 v20; // [rsp+B0h] [rbp-60h]
  unsigned __int64 v21; // [rsp+B8h] [rbp-58h]
  int v22; // [rsp+C4h] [rbp-4Ch]
  int v23; // [rsp+C8h] [rbp-48h]

  v10 = 1;
  sub_19CCCA0((__int64)v15, a3, a4, a5, a6, a7, a8, a9, a10, a1 + 153, a2);
  if ( v22 == v23 )
  {
    v12 = v16;
    if ( v17 == v16 )
    {
      v13 = &v16[v19];
      if ( v16 == v13 )
      {
        v14 = v16;
      }
      else
      {
        do
        {
          if ( *v12 == &unk_4F9EE48 )
            break;
          ++v12;
        }
        while ( v13 != v12 );
        v14 = &v16[v19];
      }
    }
    else
    {
      v13 = &v17[v18];
      v12 = (void **)sub_16CC9F0((__int64)v15, (__int64)&unk_4F9EE48);
      if ( *v12 == &unk_4F9EE48 )
      {
        if ( v17 == v16 )
          v14 = &v17[v19];
        else
          v14 = &v17[v18];
      }
      else
      {
        if ( v17 != v16 )
        {
          v12 = &v17[v18];
LABEL_11:
          LOBYTE(v10) = v13 == v12;
          goto LABEL_2;
        }
        v12 = &v17[v19];
        v14 = v12;
      }
    }
    while ( v14 != v12 && (unsigned __int64)*v12 >= 0xFFFFFFFFFFFFFFFELL )
      ++v12;
    goto LABEL_11;
  }
LABEL_2:
  if ( v21 != v20 )
    _libc_free(v21);
  if ( v17 != v16 )
    _libc_free((unsigned __int64)v17);
  j___libc_free_0(0);
  j___libc_free_0(0);
  j___libc_free_0(0);
  return v10;
}
