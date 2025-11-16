// Function: sub_B8D5C0
// Address: 0xb8d5c0
//
__int64 __fastcall sub_B8D5C0(unsigned __int8 *a1)
{
  unsigned int v1; // r12d
  int v2; // edx
  unsigned __int64 v4; // rdx
  __int64 v5; // rax

  v2 = *a1;
  LOBYTE(v1) = (unsigned __int8)(v2 - 61) <= 1u || (unsigned __int8)(v2 - 64) <= 2u;
  if ( !(_BYTE)v1 )
  {
    v4 = (unsigned int)(v2 - 34);
    if ( (unsigned __int8)v4 <= 0x33u )
    {
      v5 = 0x8000000000041LL;
      if ( _bittest64(&v5, v4) )
      {
        if ( (unsigned __int8)sub_B46420((__int64)a1) )
          return 1;
        v1 = sub_B46490((__int64)a1);
        if ( (_BYTE)v1 || (unsigned int)sub_B49D00((__int64)a1) )
          return 1;
      }
    }
  }
  return v1;
}
