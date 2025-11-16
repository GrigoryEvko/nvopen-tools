// Function: sub_3880FB0
// Address: 0x3880fb0
//
__int64 __fastcall sub_3880FB0(unsigned __int64 *a1)
{
  int v1; // ebx
  unsigned int v2; // r9d
  unsigned __int8 *i; // r15
  unsigned __int8 *v4; // r13
  unsigned __int8 v5; // bl
  int v6; // r9d
  unsigned __int8 v8; // [rsp+7h] [rbp-39h]
  _BYTE *v9; // [rsp+8h] [rbp-38h]

  v9 = (_BYTE *)*a1;
  v1 = *(unsigned __int8 *)*a1;
  LOBYTE(v2) = isalpha(v1) != 0;
  if ( (unsigned __int8)(v1 - 36) <= 0x3Bu )
    v2 |= (0x800000000000601uLL >> ((unsigned __int8)v1 - 36)) & 1;
  if ( (_BYTE)v2 )
  {
    *a1 = (unsigned __int64)(v9 + 1);
    for ( i = v9 + 1; ; *a1 = (unsigned __int64)i )
    {
      v4 = i;
      v5 = *i;
      LOBYTE(v6) = isalnum(*i) == 0;
      if ( (unsigned __int8)(v5 - 36) <= 0x3Bu )
        v6 &= ~(unsigned int)(0x800000000000601uLL >> (v5 - 36));
      ++i;
      if ( (_BYTE)v6 )
        break;
    }
    v8 = v6;
    sub_2241130(a1 + 8, 0, a1[9], v9, v4 - v9);
    return v8;
  }
  return v2;
}
