// Function: sub_16BC2E0
// Address: 0x16bc2e0
//
__int64 __fastcall sub_16BC2E0(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rdi
  int v10; // eax
  _BYTE *v11; // rsi
  size_t v13; // rdx
  int v14; // [rsp+Ch] [rbp-64h] BYREF
  void *dest; // [rsp+10h] [rbp-60h] BYREF
  size_t v16; // [rsp+18h] [rbp-58h]
  _QWORD v17[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v18; // [rsp+30h] [rbp-40h] BYREF
  size_t n; // [rsp+38h] [rbp-38h]
  _QWORD src[6]; // [rsp+40h] [rbp-30h] BYREF

  v14 = a2;
  dest = v17;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v13 = 0;
    v7 = v17;
    v18 = src;
LABEL_18:
    v16 = v13;
    *((_BYTE *)v7 + v13) = 0;
    v8 = v18;
    goto LABEL_6;
  }
  v18 = src;
  sub_16BA750((__int64 *)&v18, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = dest;
  if ( v18 == src )
  {
    v13 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v13 = n;
      v7 = dest;
    }
    goto LABEL_18;
  }
  if ( dest == v17 )
  {
    dest = v18;
    v16 = n;
    v17[0] = src[0];
  }
  else
  {
    v9 = v17[0];
    dest = v18;
    v16 = n;
    v17[0] = src[0];
    if ( v8 )
    {
      v18 = v8;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v18 = src;
  v8 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v18 != src )
    j_j___libc_free_0(v18, src[0] + 1LL);
  sub_16BBAA0(*(_QWORD *)(a1 + 160), (__int64)&dest);
  v10 = v14;
  v11 = *(_BYTE **)(a1 + 176);
  *(_DWORD *)(a1 + 16) = v14;
  if ( v11 == *(_BYTE **)(a1 + 184) )
  {
    sub_B8BBF0(a1 + 168, v11, &v14);
  }
  else
  {
    if ( v11 )
    {
      *(_DWORD *)v11 = v10;
      v11 = *(_BYTE **)(a1 + 176);
    }
    *(_QWORD *)(a1 + 176) = v11 + 4;
  }
  if ( dest != v17 )
    j_j___libc_free_0(dest, v17[0] + 1LL);
  return 0;
}
