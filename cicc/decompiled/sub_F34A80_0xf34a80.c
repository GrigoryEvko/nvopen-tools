// Function: sub_F34A80
// Address: 0xf34a80
//
__int64 __fastcall sub_F34A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // edx
  __int64 v11; // rdi
  __int64 v12; // r10
  _QWORD *v13; // rax

  result = sub_AA5930(a1);
  if ( result != v7 )
  {
    v8 = result;
    v9 = v7;
    v10 = 0;
    if ( a4 != result )
    {
      do
      {
        v11 = *(_QWORD *)(v8 - 8);
        v12 = 32LL * *(unsigned int *)(v8 + 72);
        v13 = (_QWORD *)(v11 + v12 + 8LL * v10);
        if ( a2 != *v13 )
        {
          v13 = (_QWORD *)(v11 + v12);
          v10 = 0;
          if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) != 0 )
          {
            while ( a2 != *v13 )
            {
              ++v10;
              ++v13;
              if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) == v10 )
                goto LABEL_14;
            }
          }
          else
          {
LABEL_14:
            v10 = -1;
            v13 = (_QWORD *)(v11 + v12 + 0x7FFFFFFF8LL);
          }
        }
        *v13 = a3;
        result = *(_QWORD *)(v8 + 32);
        if ( !result )
          BUG();
        v8 = 0;
        if ( *(_BYTE *)(result - 24) == 84 )
          v8 = result - 24;
      }
      while ( a4 != v8 && v8 != v9 );
    }
  }
  return result;
}
