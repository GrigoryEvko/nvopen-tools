// Function: sub_2FBF180
// Address: 0x2fbf180
//
void __fastcall sub_2FBF180(_DWORD *src, _DWORD *a2, __int64 a3)
{
  _DWORD *v3; // r8
  _DWORD *v5; // rbx
  int v6; // r12d
  _DWORD *v7; // r9
  int v8; // edx
  _DWORD *i; // rax

  if ( src != a2 )
  {
    v3 = src + 1;
    if ( a2 != src + 1 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *v3;
          v7 = v3;
          if ( *v3 != -1 )
            break;
LABEL_14:
          *v7 = v6;
          if ( a2 == ++v3 )
            return;
        }
        if ( *src != -1
          && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a3 + 8LL)
                       + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)a3 + 32LL) + v6)
                       + 8) <= *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a3 + 8LL)
                                         + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)a3 + 32LL) + *src)
                                         + 8) )
        {
          v8 = *(v3 - 1);
          for ( i = v3; ; --i )
          {
            v7 = i;
            if ( v8 != -1
              && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a3 + 8LL)
                           + 40LL * (unsigned int)(v6 + *(_DWORD *)(*(_QWORD *)a3 + 32LL))
                           + 8) <= *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a3 + 8LL)
                                             + 40LL * (unsigned int)(v8 + *(_DWORD *)(*(_QWORD *)a3 + 32LL))
                                             + 8) )
            {
              break;
            }
            *i = v8;
            v8 = *(i - 2);
          }
          goto LABEL_14;
        }
        v5 = v3 + 1;
        if ( src != v3 )
          memmove(src + 1, src, (char *)v3 - (char *)src);
        *src = v6;
        v3 = v5;
      }
      while ( a2 != v5 );
    }
  }
}
