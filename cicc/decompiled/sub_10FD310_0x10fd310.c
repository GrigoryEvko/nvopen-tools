// Function: sub_10FD310
// Address: 0x10fd310
//
__int64 __fastcall sub_10FD310(_BYTE *a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = *a1;
  if ( *a1 > 0x15u )
  {
    if ( v2 == 68 || v2 == 69 || v2 == 67 )
    {
      v5 = *((_QWORD *)a1 - 4);
      if ( v5 )
      {
        v3 = 1;
        if ( a2 == *(_QWORD *)(v5 + 8) )
          return v3;
      }
    }
    return 0;
  }
  else
  {
    v3 = 0;
    if ( v2 == 5 )
      return v3;
    return (unsigned int)sub_AD6CA0((__int64)a1) ^ 1;
  }
}
