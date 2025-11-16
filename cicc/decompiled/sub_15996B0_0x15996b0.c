// Function: sub_15996B0
// Address: 0x15996b0
//
__int64 __fastcall sub_15996B0(__int64 a1, char *a2, signed __int64 a3, char a4)
{
  __int64 v5; // rax
  __int64 **v6; // rax
  int v8; // edx
  char *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  char *v12; // r13
  size_t v13; // r12
  __int64 v14; // rax
  __int64 **v15; // rax
  __int64 v16; // r12
  void *src; // [rsp+0h] [rbp-80h] BYREF
  __int64 v18; // [rsp+8h] [rbp-78h]
  _BYTE v19[112]; // [rsp+10h] [rbp-70h] BYREF

  if ( !a4 )
  {
    v5 = sub_1644C60(a1, 8);
    v6 = (__int64 **)sub_1645D80(v5, a3);
    return sub_15991C0(a2, a3, v6);
  }
  v8 = 0;
  src = v19;
  v9 = v19;
  v18 = 0x4000000000LL;
  if ( (unsigned __int64)a3 > 0x40 )
  {
    sub_16CD150(&src, v19, a3, 1);
    v8 = v18;
    v9 = (char *)src + (unsigned int)v18;
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
      v8 = v18;
    }
  }
  else if ( a3 > 0 )
  {
    goto LABEL_5;
  }
  LODWORD(v18) = v8 + a3;
  v11 = (unsigned int)(v8 + a3);
  if ( HIDWORD(v18) <= (unsigned int)(v8 + a3) )
  {
    sub_16CD150(&src, v19, 0, 1);
    v11 = (unsigned int)v18;
  }
  *((_BYTE *)src + v11) = 0;
  v12 = (char *)src;
  LODWORD(v18) = v18 + 1;
  v13 = (unsigned int)v18;
  v14 = sub_1644C60(a1, 8);
  v15 = (__int64 **)sub_1645D80(v14, v13);
  v16 = sub_15991C0(v12, v13, v15);
  if ( src != v19 )
    _libc_free((unsigned __int64)src);
  return v16;
}
