// Function: sub_16E8DE0
// Address: 0x16e8de0
//
__int64 __fastcall sub_16E8DE0(unsigned __int8 **a1)
{
  unsigned __int8 *v1; // rax
  unsigned __int8 *v2; // r9
  int v3; // ecx
  int v4; // r8d

  v1 = *a1;
  v2 = a1[1];
  v3 = 0;
  v4 = 0;
  if ( *a1 < v2 )
  {
    do
    {
      if ( (unsigned int)*v1 - 48 > 9 )
        break;
      if ( v4 > 255 )
        goto LABEL_9;
      ++v1;
      ++v3;
      *a1 = v1;
      v4 = (char)*(v1 - 1) + 10 * v4 - 48;
    }
    while ( v1 != v2 );
    if ( v3 > 0 && v4 <= 255 )
      return (unsigned int)v4;
  }
LABEL_9:
  if ( !*((_DWORD *)a1 + 4) )
    *((_DWORD *)a1 + 4) = 10;
  *a1 = (unsigned __int8 *)&unk_4FA17D0;
  a1[1] = (unsigned __int8 *)&unk_4FA17D0;
  return (unsigned int)v4;
}
