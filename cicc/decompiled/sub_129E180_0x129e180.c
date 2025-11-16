// Function: sub_129E180
// Address: 0x129e180
//
__int64 __fastcall sub_129E180(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rdx
  int v6; // eax
  char *v7; // rax
  int v8; // edx
  __int64 v9; // rbx
  int v10; // eax
  int v11; // [rsp+1Ch] [rbp-14h] BYREF

  if ( !memcmp((const void *)a1, "__nv_static_", 0xCu) )
  {
    v9 = a1 + 12;
    v11 = 0;
    sscanf((const char *)(a1 + 12), "%d", &v11);
    if ( (unsigned int)(*(char *)(a1 + 12) - 48) <= 9 )
    {
      do
        v10 = *(char *)++v9;
      while ( (unsigned int)(v10 - 48) <= 9 );
    }
    return v9 + v11 + 2;
  }
  else
  {
    result = a1;
    if ( !memcmp((const void *)a1, "__cuda_local_var_", 0x11u) )
    {
      v5 = a1 + 17;
      if ( (unsigned int)(*(char *)(a1 + 17) - 48) <= 9 )
      {
        do
          v6 = *(char *)++v5;
        while ( (unsigned int)(v6 - 48) <= 9 );
      }
      v7 = (char *)(v5 + 1);
      if ( (unsigned int)(*(char *)(v5 + 1) - 48) <= 9 )
      {
        do
          v8 = *++v7;
        while ( (unsigned int)(v8 - 48) <= 9 );
      }
      if ( !memcmp(v7, "_const_", 7u) )
      {
        return (__int64)(v7 + 7);
      }
      else
      {
        if ( memcmp(v7, "_non_const_", 0xBu) )
          sub_127B550("cannot demangle cudafe mangled name!", (_DWORD *)(a2 + 64), 1);
        return (__int64)(v7 + 11);
      }
    }
  }
  return result;
}
