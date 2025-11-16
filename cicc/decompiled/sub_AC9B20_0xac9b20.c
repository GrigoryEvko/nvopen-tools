// Function: sub_AC9B20
// Address: 0xac9b20
//
__int64 __fastcall sub_AC9B20(__int64 a1, char *a2, signed __int64 a3, char a4)
{
  __int64 v5; // rax
  __int64 **v6; // rax
  __int64 v8; // rdx
  char *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r12
  char *v12; // r13
  size_t v13; // r12
  __int64 v14; // rax
  __int64 **v15; // rax
  size_t v16; // rsi
  __int64 v17; // r12
  void *src; // [rsp+0h] [rbp-90h] BYREF
  __int64 v19; // [rsp+8h] [rbp-88h]
  unsigned __int64 v20; // [rsp+10h] [rbp-80h]
  _BYTE v21[120]; // [rsp+18h] [rbp-78h] BYREF

  if ( !a4 )
  {
    v5 = sub_BCD140(a1, 8);
    v6 = (__int64 **)sub_BCD420(v5, a3);
    return sub_AC9630(a2, a3, v6);
  }
  v8 = 0;
  v19 = 0;
  src = v21;
  v9 = v21;
  v20 = 64;
  if ( (unsigned __int64)a3 > 0x40 )
  {
    sub_C8D290(&src, v21, a3, 1);
    v8 = v19;
    v9 = (char *)src + v19;
    if ( a3 > 0 )
    {
LABEL_5:
      v10 = 0;
      do
      {
        v9[v10] = a2[v10];
        ++v10;
      }
      while ( a3 != v10 );
      v8 = v19;
    }
  }
  else if ( a3 > 0 )
  {
    goto LABEL_5;
  }
  v11 = v8 + a3;
  v19 = v11;
  if ( v11 + 1 > v20 )
  {
    sub_C8D290(&src, v21, v11 + 1, 1);
    v11 = v19;
  }
  *((_BYTE *)src + v11) = 0;
  v12 = (char *)src;
  v13 = ++v19;
  v14 = sub_BCD140(a1, 8);
  v15 = (__int64 **)sub_BCD420(v14, v13);
  v16 = v13;
  v17 = sub_AC9630(v12, v13, v15);
  if ( src != v21 )
    _libc_free(src, v16);
  return v17;
}
