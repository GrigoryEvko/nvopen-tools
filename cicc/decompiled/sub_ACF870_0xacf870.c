// Function: sub_ACF870
// Address: 0xacf870
//
__int64 __fastcall sub_ACF870(__int64 *a1)
{
  __int64 *v1; // rdx
  __int64 *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // ebx
  int v8; // [rsp+Ch] [rbp-164h] BYREF
  __int64 v9[4]; // [rsp+10h] [rbp-160h] BYREF
  __int64 *v10; // [rsp+30h] [rbp-140h] BYREF
  __int64 v11; // [rsp+38h] [rbp-138h]
  _BYTE v12[304]; // [rsp+40h] [rbp-130h] BYREF

  v1 = (__int64 *)v12;
  v2 = a1 - 12;
  v10 = (__int64 *)v12;
  v3 = *(a1 - 16);
  v11 = 0x2000000000LL;
  v4 = 0;
  while ( 1 )
  {
    v1[v4] = v3;
    v4 = (unsigned int)(v11 + 1);
    LODWORD(v11) = v11 + 1;
    if ( a1 == v2 )
      break;
    v3 = *v2;
    if ( v4 + 1 > (unsigned __int64)HIDWORD(v11) )
    {
      sub_C8D5F0(&v10, v12, v4 + 1, 8);
      v4 = (unsigned int)v11;
    }
    v1 = v10;
    v2 += 4;
  }
  v5 = a1[1];
  v9[2] = v4;
  v9[0] = v5;
  v9[1] = (__int64)v10;
  v8 = sub_AC5F60(v10, (__int64)&v10[v4]);
  v6 = sub_AC7AE0(v9, &v8);
  if ( v10 != (__int64 *)v12 )
    _libc_free(v10, &v8);
  return v6;
}
