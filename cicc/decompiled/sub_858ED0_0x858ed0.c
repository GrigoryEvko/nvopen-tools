// Function: sub_858ED0
// Address: 0x858ed0
//
__int64 **__fastcall sub_858ED0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 **v4; // r13
  __int64 *v5; // rbx
  unsigned __int64 v8; // rcx
  size_t v9; // rdx
  int v10; // eax
  __int64 v12; // [rsp+8h] [rbp-38h]

  v4 = *(__int64 ***)(*a1 + 8 * a2);
  if ( v4 )
  {
    v5 = *v4;
    v8 = (*v4)[5];
    while ( 1 )
    {
      if ( v8 == a4 )
      {
        v9 = *(_QWORD *)(a3 + 8);
        if ( v9 == v5[2] )
        {
          if ( !v9 )
            break;
          v12 = a3;
          v10 = memcmp(*(const void **)a3, (const void *)v5[1], v9);
          a3 = v12;
          if ( !v10 )
            break;
        }
      }
      if ( !*v5 )
        return 0;
      v8 = *(_QWORD *)(*v5 + 40);
      v4 = (__int64 **)v5;
      if ( a2 != v8 % a1[1] )
        return 0;
      v5 = (__int64 *)*v5;
    }
  }
  return v4;
}
