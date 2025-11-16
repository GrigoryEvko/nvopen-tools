// Function: sub_16CCD50
// Address: 0x16ccd50
//
__int64 __fastcall sub_16CCD50(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-28h]

  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a2 + 16) == *(_QWORD *)(a2 + 8) )
  {
    if ( v7 != v8 )
    {
      _libc_free(v7);
      v8 = *(_QWORD *)(a1 + 8);
    }
    *(_QWORD *)(a1 + 16) = v8;
  }
  else
  {
    v9 = *(unsigned int *)(a2 + 24);
    if ( *(_DWORD *)(a1 + 24) != (_DWORD)v9 )
    {
      v10 = 8 * v9;
      if ( v7 == v8 )
      {
        v13 = malloc(8 * v9);
        if ( !v13 )
        {
          if ( v10 || (v14 = malloc(1u)) == 0 )
            sub_16BD1C0("Allocation failed", 1u);
          else
            v13 = v14;
        }
        *(_QWORD *)(a1 + 16) = v13;
      }
      else
      {
        v11 = (__int64)realloc(v7, 8 * v9, v8, a4, a5, a6);
        if ( !v11 && (v10 || (v11 = malloc(1u)) == 0) )
        {
          v15 = v11;
          sub_16BD1C0("Allocation failed", 1u);
          v11 = v15;
        }
        *(_QWORD *)(a1 + 16) = v11;
      }
    }
  }
  return sub_16CCC50(a1, a2);
}
