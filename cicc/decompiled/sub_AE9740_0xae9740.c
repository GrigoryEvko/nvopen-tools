// Function: sub_AE9740
// Address: 0xae9740
//
__int64 __fastcall sub_AE9740(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  _BYTE *v4; // rdx
  _BYTE *v5; // r8
  _BYTE *v6; // r9
  size_t v7; // r10
  __int64 v8; // r13
  __int64 *v9; // rdi
  int v10; // eax
  __int64 *v11; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // [rsp+8h] [rbp-88h]
  size_t v17; // [rsp+10h] [rbp-80h]
  _BYTE *v18; // [rsp+18h] [rbp-78h]
  __int64 *v19; // [rsp+20h] [rbp-70h] BYREF
  __int64 v20; // [rsp+28h] [rbp-68h]
  _BYTE dest[96]; // [rsp+30h] [rbp-60h] BYREF

  v3 = a2;
  v19 = (__int64 *)dest;
  v5 = (_BYTE *)sub_AE9410(a1);
  v6 = v4;
  v7 = v4 - v5;
  v20 = 0x600000000LL;
  v8 = (v4 - v5) >> 3;
  if ( (unsigned __int64)(v4 - v5) > 0x30 )
  {
    v18 = v4;
    a2 = (__int64)dest;
    v16 = v5;
    v17 = v4 - v5;
    sub_C8D5F0(&v19, dest, (v4 - v5) >> 3, 8);
    v11 = v19;
    v10 = v20;
    v6 = v18;
    v7 = v17;
    v5 = v16;
    v9 = &v19[(unsigned int)v20];
  }
  else
  {
    v9 = (__int64 *)dest;
    v10 = 0;
    v11 = (__int64 *)dest;
  }
  if ( v5 != v6 )
  {
    a2 = (__int64)v5;
    memcpy(v9, v5, v7);
    v11 = v19;
    v10 = v20;
  }
  LODWORD(v20) = v8 + v10;
  result = (unsigned int)(v8 + v10);
  for ( i = &v11[result]; i != v11; result = sub_B99FD0(v14, 38, v3) )
  {
    v14 = *v11;
    a2 = 38;
    ++v11;
  }
  v15 = *(_QWORD *)(a1 + 8);
  if ( (v15 & 4) != 0 )
  {
    a2 = v3;
    result = sub_BA6110(v15 & 0xFFFFFFFFFFFFFFF8LL, v3);
  }
  if ( v19 != (__int64 *)dest )
    return _libc_free(v19, a2);
  return result;
}
