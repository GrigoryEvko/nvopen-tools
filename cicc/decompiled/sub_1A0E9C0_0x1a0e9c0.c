// Function: sub_1A0E9C0
// Address: 0x1a0e9c0
//
__int64 __fastcall sub_1A0E9C0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        __m128 a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r14d
  double v11; // xmm4_8
  double v12; // xmm5_8
  void **v14; // rax
  void **v15; // rbx
  void **v16; // rdx
  char v17[8]; // [rsp+70h] [rbp-A0h] BYREF
  void **v18; // [rsp+78h] [rbp-98h]
  void **v19; // [rsp+80h] [rbp-90h]
  int v20; // [rsp+88h] [rbp-88h]
  int v21; // [rsp+8Ch] [rbp-84h]
  __int64 v22; // [rsp+B0h] [rbp-60h]
  unsigned __int64 v23; // [rsp+B8h] [rbp-58h]
  int v24; // [rsp+C4h] [rbp-4Ch]
  int v25; // [rsp+C8h] [rbp-48h]

  v10 = 0;
  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return v10;
  v10 = 1;
  sub_1A0D7B0((__int64)v17, a1 + 160, a2, a3, a4, a5, a6, v11, v12, a9, a10);
  if ( v24 == v25 )
  {
    v14 = v18;
    if ( v19 == v18 )
    {
      v15 = &v18[v21];
      if ( v18 == v15 )
      {
        v16 = v18;
      }
      else
      {
        do
        {
          if ( *v14 == &unk_4F9EE48 )
            break;
          ++v14;
        }
        while ( v15 != v14 );
        v16 = &v18[v21];
      }
    }
    else
    {
      v15 = &v19[v20];
      v14 = (void **)sub_16CC9F0((__int64)v17, (__int64)&unk_4F9EE48);
      if ( *v14 == &unk_4F9EE48 )
      {
        if ( v19 == v18 )
          v16 = &v19[v21];
        else
          v16 = &v19[v20];
      }
      else
      {
        if ( v19 != v18 )
        {
          v14 = &v19[v20];
LABEL_13:
          LOBYTE(v10) = v15 == v14;
          goto LABEL_4;
        }
        v14 = &v19[v21];
        v16 = v14;
      }
    }
    while ( v16 != v14 && (unsigned __int64)*v14 >= 0xFFFFFFFFFFFFFFFELL )
      ++v14;
    goto LABEL_13;
  }
LABEL_4:
  if ( v23 != v22 )
    _libc_free(v23);
  if ( v19 != v18 )
    _libc_free((unsigned __int64)v19);
  j___libc_free_0(0);
  j___libc_free_0(0);
  j___libc_free_0(0);
  return v10;
}
