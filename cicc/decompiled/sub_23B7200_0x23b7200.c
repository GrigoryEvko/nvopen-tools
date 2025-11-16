// Function: sub_23B7200
// Address: 0x23b7200
//
__int64 __fastcall sub_23B7200(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // eax
  __int64 v6; // r9
  unsigned int v7; // esi
  _QWORD *v8; // rdi
  __int64 v9; // r8
  _QWORD *v10; // r10
  int v11; // ebx

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v4 = *a2;
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v5 & (37 * *a2);
    v8 = (_QWORD *)(v6 + 40LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
      *a3 = v8;
      return 1;
    }
    else
    {
      v10 = 0;
      v11 = 1;
      while ( v9 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( !v10 && v9 == 0x7FFFFFFFFFFFFFFELL )
          v10 = v8;
        v7 = v5 & (v11 + v7);
        v8 = (_QWORD *)(v6 + 40LL * v7);
        v9 = *v8;
        if ( *v8 == v4 )
        {
          *a3 = v8;
          return 1;
        }
        ++v11;
      }
      if ( !v10 )
        v10 = v8;
      result = 0;
      *a3 = v10;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
