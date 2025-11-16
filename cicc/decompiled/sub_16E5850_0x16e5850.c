// Function: sub_16E5850
// Address: 0x16e5850
//
__int64 __fastcall sub_16E5850(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rcx
  size_t v8; // rsi
  __int64 v9; // rdi
  size_t v11; // rdx
  _QWORD *v12; // [rsp+0h] [rbp-30h] BYREF
  size_t n; // [rsp+8h] [rbp-28h]
  _QWORD src[4]; // [rsp+10h] [rbp-20h] BYREF

  if ( !a1 )
  {
    LOBYTE(src[0]) = 0;
    v5 = *(_BYTE **)a4;
    v11 = 0;
    v12 = src;
LABEL_12:
    *(_QWORD *)(a4 + 8) = v11;
    v5[v11] = 0;
    v6 = v12;
    goto LABEL_6;
  }
  v12 = src;
  sub_16E3680((__int64 *)&v12, a1, (__int64)&a1[a2]);
  v5 = *(_BYTE **)a4;
  v6 = *(_QWORD **)a4;
  if ( v12 == src )
  {
    v11 = n;
    if ( n )
    {
      if ( n == 1 )
        *v5 = src[0];
      else
        memcpy(v5, src, n);
      v11 = n;
      v5 = *(_BYTE **)a4;
    }
    goto LABEL_12;
  }
  v7 = src[0];
  v8 = n;
  if ( v5 == (_BYTE *)(a4 + 16) )
  {
    *(_QWORD *)a4 = v12;
    *(_QWORD *)(a4 + 8) = v8;
    *(_QWORD *)(a4 + 16) = v7;
  }
  else
  {
    v9 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)a4 = v12;
    *(_QWORD *)(a4 + 8) = v8;
    *(_QWORD *)(a4 + 16) = v7;
    if ( v6 )
    {
      v12 = v6;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v12 = src;
  v6 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v6 = 0;
  if ( v12 != src )
    j_j___libc_free_0(v12, src[0] + 1LL);
  return 0;
}
