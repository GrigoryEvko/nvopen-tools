// Function: sub_CB7460
// Address: 0xcb7460
//
__int64 __fastcall sub_CB7460(unsigned __int8 **a1)
{
  unsigned __int8 *v1; // r9
  unsigned __int8 *v2; // rax
  int v3; // ecx
  int v4; // r8d

  v1 = a1[1];
  v2 = *a1;
  v3 = 0;
  v4 = 0;
  if ( v1 - *a1 > 0 )
  {
    do
    {
      if ( (unsigned int)*v2 - 48 > 9 )
        break;
      if ( v4 > 255 )
        goto LABEL_9;
      ++v2;
      ++v3;
      *a1 = v2;
      v4 = (char)*(v2 - 1) + 10 * v4 - 48;
    }
    while ( v1 - v2 > 0 );
    if ( v3 > 0 && v4 <= 255 )
      return (unsigned int)v4;
  }
LABEL_9:
  if ( !*((_DWORD *)a1 + 4) )
    *((_DWORD *)a1 + 4) = 10;
  *a1 = (unsigned __int8 *)&unk_4F85140;
  a1[1] = (unsigned __int8 *)&unk_4F85140;
  return (unsigned int)v4;
}
