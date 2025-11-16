// Function: sub_15C80C0
// Address: 0x15c80c0
//
__int64 __fastcall sub_15C80C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // r13
  size_t v7; // rdx
  __int64 v8; // r12
  __int64 (__fastcall *v9)(__int64, __int64, size_t); // r14
  unsigned int v10; // r12d
  char *v11; // r15
  size_t v12; // rax
  const void *v13; // r14
  size_t v14; // r13
  size_t v15; // rax

  v2 = sub_15E0530(*(_QWORD *)(a1 + 16));
  v5 = sub_16033E0(v2, a2, v3, v4);
  v6 = *(_QWORD *)(a1 + 48);
  v7 = 0;
  v8 = v5;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, size_t))(*(_QWORD *)v5 + 24LL);
  if ( v6 )
    v7 = strlen(*(const char **)(a1 + 48));
  v10 = v9(v8, v6, v7);
  if ( !(_BYTE)v10 )
  {
    v11 = (char *)off_4C6F360;
    if ( off_4C6F360 )
    {
      v12 = strlen((const char *)off_4C6F360);
      v13 = *(const void **)(a1 + 48);
      v14 = v12;
      v15 = 0;
      if ( !v13 )
      {
LABEL_7:
        if ( v15 != v14 )
          return v10;
        if ( v15 )
        {
          LOBYTE(v10) = memcmp(v13, v11, v15) == 0;
          return v10;
        }
        return 1;
      }
    }
    else
    {
      v13 = *(const void **)(a1 + 48);
      v14 = 0;
      if ( !v13 )
        return 1;
    }
    v15 = strlen((const char *)v13);
    goto LABEL_7;
  }
  return v10;
}
