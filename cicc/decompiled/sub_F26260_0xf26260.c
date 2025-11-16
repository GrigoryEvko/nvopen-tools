// Function: sub_F26260
// Address: 0xf26260
//
__int64 __fastcall sub_F26260(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  unsigned int i; // ebx
  __int64 v6; // rdx
  __int64 result; // rax
  int v8; // [rsp+14h] [rbp-7Ch]
  _QWORD v10[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v11[96]; // [rsp+30h] [rbp-60h] BYREF

  v10[0] = v11;
  v10[1] = 0x600000000LL;
  v3 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 != a2 + 48 )
  {
    if ( !v3 )
      BUG();
    v4 = v3 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 <= 0xA )
    {
      v8 = sub_B46E30(v4);
      if ( v8 )
      {
        for ( i = 0; i != v8; ++i )
        {
          v6 = sub_B46EC0(v4, i);
          if ( a3 != v6 )
            sub_F25920(a1, a2, v6, (__int64)v10);
        }
      }
    }
  }
  result = sub_F26140(a1, (__int64)v10);
  if ( (_BYTE *)v10[0] != v11 )
    return _libc_free(v10[0], v10);
  return result;
}
