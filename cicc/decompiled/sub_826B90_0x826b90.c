// Function: sub_826B90
// Address: 0x826b90
//
__int64 __fastcall sub_826B90(__int64 **a1, __int64 **a2, unsigned int a3, _DWORD *a4, _QWORD *a5)
{
  __int64 **v5; // r15
  __int64 **v9; // rbx
  const char *v10; // r8
  __int64 result; // rax
  char v12; // dl

  if ( a2 )
  {
    v5 = a1;
    v9 = a2;
    if ( a1 )
    {
      while ( 1 )
      {
        v12 = (_BYTE)v9[4] & 2;
        if ( ((_BYTE)v5[4] & 2) == 0 )
          break;
        if ( !v12 )
          goto LABEL_5;
        v5 = (__int64 **)*v5;
        v9 = (__int64 **)*v9;
        if ( !v5 )
          return result;
LABEL_9:
        if ( !v9 )
          return result;
      }
      if ( v12 )
      {
LABEL_5:
        v10 = (const char *)v9[3];
        if ( !v10 )
          v10 = byte_3F871B3;
        result = sub_6861C0(7u, a3, a4, a5, (__int64)v10);
      }
      v5 = (__int64 **)*v5;
      v9 = (__int64 **)*v9;
      if ( !v5 )
        return result;
      goto LABEL_9;
    }
  }
  return result;
}
