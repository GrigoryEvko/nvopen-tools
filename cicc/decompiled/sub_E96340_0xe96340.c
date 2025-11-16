// Function: sub_E96340
// Address: 0xe96340
//
unsigned __int64 __fastcall sub_E96340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  int v7; // r12d
  unsigned __int64 v8; // r13
  __int64 v9; // rdx
  __int64 i; // rdx

  result = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL);
  if ( result )
  {
    v7 = *(_DWORD *)(result + 24);
    result = *(unsigned int *)(a1 + 184);
    v8 = (unsigned int)(v7 + 1);
    if ( v8 != result )
    {
      if ( v8 >= result )
      {
        if ( v8 > *(unsigned int *)(a1 + 188) )
        {
          sub_C8D5F0(a1 + 176, (const void *)(a1 + 192), v8, 8u, a5, a6);
          result = *(unsigned int *)(a1 + 184);
        }
        v9 = *(_QWORD *)(a1 + 176);
        result = v9 + 8 * result;
        for ( i = v9 + 8 * v8; i != result; result += 8LL )
        {
          if ( result )
            *(_QWORD *)result = 0;
        }
      }
      *(_DWORD *)(a1 + 184) = v8;
    }
  }
  return result;
}
