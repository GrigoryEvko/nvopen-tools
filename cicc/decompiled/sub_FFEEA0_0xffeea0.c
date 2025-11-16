// Function: sub_FFEEA0
// Address: 0xffeea0
//
__int64 __fastcall sub_FFEEA0(_BYTE *a1, unsigned int *a2, __int64 a3)
{
  int v3; // r13d
  _BYTE *v4; // r12
  __int64 v6; // r14
  unsigned int v7; // ebx
  __int64 v8; // rdx
  size_t v9; // rdx

  v3 = a3;
  v4 = a1;
  if ( *a1 <= 0x15u )
    return sub_AAADB0((__int64)a1, a2, a3);
  if ( *a1 == 94 )
  {
    v6 = (unsigned int)a3;
    while ( 1 )
    {
      v7 = *((_DWORD *)v4 + 20);
      v8 = v7;
      if ( (unsigned int)v6 <= v7 )
        v8 = v6;
      v9 = 4 * v8;
      if ( !v9 || !memcmp(*((const void **)v4 + 9), a2, v9) )
        break;
      v4 = (_BYTE *)*((_QWORD *)v4 - 8);
      if ( !v4 )
        BUG();
      if ( *v4 != 94 )
        return 0;
    }
    if ( v3 == v7 )
      return *((_QWORD *)v4 - 4);
  }
  return 0;
}
