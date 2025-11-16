// Function: sub_29BF2A0
// Address: 0x29bf2a0
//
void __fastcall sub_29BF2A0(_QWORD *src, _QWORD *a2, _QWORD *a3)
{
  _QWORD *i; // r12
  __int64 v7; // r15
  _QWORD *v8; // r9
  __int64 v9; // rdi
  unsigned int v10; // esi
  __int64 v11; // rdx
  _QWORD *v12; // rax

  if ( src != a2 )
  {
    for ( i = src + 1; i != a2; *src = v7 )
    {
      while ( 1 )
      {
        v7 = *i;
        v8 = i;
        v9 = 16LL * *i;
        v10 = *(_DWORD *)(*a3 + v9);
        if ( v10 < *(_DWORD *)(*a3 + 16LL * *src) )
          break;
        v11 = *(i - 1);
        v12 = i - 1;
        if ( v10 < *(_DWORD *)(*a3 + 16 * v11) )
        {
          do
          {
            v12[1] = v11;
            v8 = v12;
            v11 = *--v12;
          }
          while ( *(_DWORD *)(*a3 + v9) < *(_DWORD *)(*a3 + 16 * v11) );
        }
        ++i;
        *v8 = v7;
        if ( i == a2 )
          return;
      }
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      ++i;
    }
  }
}
