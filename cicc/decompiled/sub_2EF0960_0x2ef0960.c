// Function: sub_2EF0960
// Address: 0x2ef0960
//
__int64 __fastcall sub_2EF0960(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax

  v4 = a3 & 0xFFFFFFFFFFFFFFF9LL;
  if ( (a2 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
  {
    if ( v4 && (a3 & 4) != 0 )
      goto LABEL_10;
    return 1;
  }
  if ( (a2 & 4) != 0 )
  {
    if ( v4 && (a3 & 4) != 0 )
    {
      if ( (unsigned __int16)((unsigned int)a3 >> 8) != (unsigned __int16)((unsigned int)a2 >> 8)
        || (((unsigned __int8)(a3 >> 3) ^ (unsigned __int8)(a2 >> 3)) & 1) != 0 )
      {
        sub_2EF06E0(a1, "operand types must preserve number of vector elements", a4);
        return 0;
      }
      return 1;
    }
LABEL_10:
    sub_2EF06E0(a1, "operand types must be all-vector or all-scalar", a4);
    return 0;
  }
  if ( !v4 )
    return 1;
  if ( (a3 & 4) != 0 )
    goto LABEL_10;
  return 1;
}
