// Function: sub_1E0A7F0
// Address: 0x1e0a7f0
//
__int64 __fastcall sub_1E0A7F0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned int v11; // r9d
  _QWORD *v12; // rdx

  v4 = a2;
  v7 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24 * v4);
  v8 = *v7;
  v9 = (v7[1] - *v7) >> 3;
  if ( v9 )
  {
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      v12 = (_QWORD *)(v8 + 8 * v10);
      if ( *v12 == a3 )
      {
        ++v10;
        *v12 = a4;
        v11 = 1;
        if ( v10 == v9 )
          return v11;
      }
      else if ( ++v10 == v9 )
      {
        return v11;
      }
      v8 = *v7;
    }
  }
  return 0;
}
