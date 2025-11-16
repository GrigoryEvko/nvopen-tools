// Function: sub_1E85E90
// Address: 0x1e85e90
//
unsigned __int64 __fastcall sub_1E85E90(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rax
  char v2; // cl
  unsigned __int64 v3; // rdx
  unsigned __int16 v4; // ax
  unsigned __int64 v6; // rax

  v1 = *a1;
  v2 = *(_BYTE *)a1 & 2;
  if ( (*(_BYTE *)a1 & 1) != 0 )
  {
    v3 = v1 >> 18;
    if ( v2 )
    {
      v3 = (unsigned __int16)v3;
      v4 = v1 >> 34;
    }
    else
    {
      v4 = v1 >> 18;
      v3 = (unsigned __int16)(*a1 >> 2);
    }
    return 4 * (v3 | ((unsigned __int64)v4 << 16)) + 1;
  }
  else
  {
    v6 = v1 >> 18;
    if ( !v2 )
      LODWORD(v6) = *a1 >> 2;
    return 4LL * (unsigned int)v6;
  }
}
