// Function: sub_824150
// Address: 0x824150
//
int __fastcall sub_824150(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  int result; // eax
  size_t v5; // rdx

  if ( a1 == a2 )
    return 1;
  v2 = a2;
  if ( a2 )
  {
    v3 = a1;
    if ( a1 )
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(*v3 + 16LL);
        if ( v5 != *(_QWORD *)(*v2 + 16LL) )
          break;
        result = strncmp(*(const char **)(*v3 + 8LL), *(const char **)(*v2 + 8LL), v5);
        if ( result )
          break;
        v2 = (_QWORD *)v2[1];
        v3 = (_QWORD *)v3[1];
        if ( v2 == v3 )
          return 1;
        if ( !v3 || !v2 )
          return result;
      }
    }
  }
  return 0;
}
