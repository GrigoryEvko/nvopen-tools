// Function: sub_2207E50
// Address: 0x2207e50
//
size_t __fastcall sub_2207E50(FILE **a1, char *a2, size_t a3, __int64 a4, __int64 a5)
{
  size_t v5; // r15
  size_t v7; // r14
  int v8; // r12d
  size_t v9; // rbx
  size_t v10; // rdx
  ssize_t v11; // rax
  size_t v13; // r15
  char *v14; // r13
  size_t v15; // rbp
  ssize_t v16; // rax
  struct iovec iovec; // [rsp+10h] [rbp-58h] BYREF
  __int64 v20; // [rsp+20h] [rbp-48h]
  __int64 v21; // [rsp+28h] [rbp-40h]

  v5 = a3;
  v7 = a3 + a5;
  v20 = a4;
  v21 = a5;
  v8 = sub_2207D30(a1);
  v9 = v7;
  while ( 1 )
  {
    while ( 1 )
    {
      iovec.iov_base = a2;
      iovec.iov_len = v5;
      v11 = writev(v8, &iovec, 2);
      if ( v11 != -1 )
        break;
      if ( *__errno_location() != 4 )
      {
        v7 -= v9;
        return v7;
      }
    }
    v9 -= v11;
    if ( !v9 )
      return v7;
    v10 = v11 - v5;
    if ( (__int64)(v11 - v5) >= 0 )
      break;
    a2 += v11;
    v5 -= v11;
  }
  v13 = a5 - v10;
  v14 = (char *)(v10 + a4);
  v15 = a5 - v10;
  do
  {
    while ( 1 )
    {
      v16 = write(v8, v14, v15);
      if ( v16 == -1 )
        break;
      v15 -= v16;
      if ( !v15 )
        goto LABEL_15;
      v14 += v16;
    }
  }
  while ( *__errno_location() == 4 );
  v13 -= v15;
LABEL_15:
  v7 -= v9 - v13;
  return v7;
}
