// Function: sub_AC7F30
// Address: 0xac7f30
//
__int64 __fastcall sub_AC7F30(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  int v5; // eax
  __int64 v6; // r10
  unsigned int v7; // r9d
  _QWORD *v8; // r8
  __int64 v9; // rdx
  _QWORD *v10; // r12
  int i; // ebx
  __int64 v12; // rdi

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v5 & *(_DWORD *)a2;
    v8 = (_QWORD *)(v6 + 8LL * v7);
    v9 = *v8;
    if ( *v8 == -4096 )
    {
      *a3 = v8;
      return 0;
    }
    else
    {
      v10 = 0;
      for ( i = 1; ; ++i )
      {
        if ( v9 == -8192 )
        {
          if ( !v10 )
            v10 = v8;
        }
        else if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)(v9 + 8) && *(_QWORD *)(a2 + 24) == 4 )
        {
          v12 = 0;
          while ( *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v12) == *(_QWORD *)(v9 + 32 * v12 - 128) )
          {
            if ( ++v12 == 4 )
            {
              *a3 = v8;
              return 1;
            }
          }
        }
        v7 = v5 & (i + v7);
        v8 = (_QWORD *)(v6 + 8LL * v7);
        v9 = *v8;
        if ( *v8 == -4096 )
          break;
      }
      if ( v10 )
        v8 = v10;
      *a3 = v8;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
