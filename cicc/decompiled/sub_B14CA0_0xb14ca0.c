// Function: sub_B14CA0
// Address: 0xb14ca0
//
__int64 __fastcall sub_B14CA0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // r13
  size_t v4; // rdx
  __int64 v5; // r12
  __int64 (__fastcall *v6)(__int64, __int64, size_t); // r14
  unsigned int v7; // r12d
  char *v9; // r14
  size_t v10; // rax
  const void *v11; // r15
  size_t v12; // r13
  const char *v13; // rdi

  v1 = sub_B2BE50(*(_QWORD *)(a1 + 16));
  v2 = sub_B6F970(v1);
  v3 = *(_QWORD *)(a1 + 40);
  v4 = 0;
  v5 = v2;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, size_t))(*(_QWORD *)v2 + 24LL);
  if ( v3 )
    v4 = strlen(*(const char **)(a1 + 40));
  v7 = v6(v5, v3, v4);
  if ( !(_BYTE)v7 )
  {
    v9 = (char *)off_4B91160;
    if ( off_4B91160 )
    {
      v10 = strlen((const char *)off_4B91160);
      v11 = *(const void **)(a1 + 40);
      v12 = v10;
      if ( v11 )
      {
        if ( v10 != strlen(*(const char **)(a1 + 40)) )
          return v7;
        if ( v12 )
        {
          LOBYTE(v7) = memcmp(v11, v9, v12) == 0;
          return v7;
        }
      }
      else if ( v10 )
      {
        return v7;
      }
    }
    else
    {
      v13 = *(const char **)(a1 + 40);
      if ( v13 && strlen(v13) )
        return v7;
    }
    return 1;
  }
  return v7;
}
