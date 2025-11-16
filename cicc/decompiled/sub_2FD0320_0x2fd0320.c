// Function: sub_2FD0320
// Address: 0x2fd0320
//
__int64 *__fastcall sub_2FD0320(_QWORD *a1, int *a2)
{
  unsigned __int64 v3; // rdi
  int v4; // r9d
  __int64 **v5; // r8
  __int64 *v6; // rax
  int v7; // ecx
  __int64 *result; // rax

  v3 = a1[1];
  v4 = *a2;
  v5 = *(__int64 ***)(*a1 + 8 * (*a2 % v3));
  if ( v5 )
  {
    v6 = *v5;
    if ( v4 == *((_DWORD *)*v5 + 2) )
    {
LABEL_6:
      result = *v5;
      if ( *v5 )
        return result;
    }
    else
    {
      while ( *v6 )
      {
        v7 = *(_DWORD *)(*v6 + 8);
        v5 = (__int64 **)v6;
        if ( *a2 % v3 != v7 % v3 )
          break;
        v6 = (__int64 *)*v6;
        if ( v4 == v7 )
          goto LABEL_6;
      }
    }
  }
  return 0;
}
