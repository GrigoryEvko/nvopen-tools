// Function: sub_11FF110
// Address: 0x11ff110
//
__int64 __fastcall sub_11FF110(unsigned __int8 **a1)
{
  __int64 v1; // rax
  char *v2; // r8
  char *v3; // rax
  unsigned __int64 v4; // r13

  v1 = (__int64)*a1;
  if ( (unsigned int)**a1 - 48 > 9 )
    JUMPOUT(0x11FF100);
  v2 = (char *)(v1 + 1);
  *a1 = (unsigned __int8 *)(v1 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v1 + 1) - 48 <= 9 )
  {
    v3 = (char *)(v1 + 2);
    do
    {
      v2 = v3;
      *a1 = (unsigned __int8 *)v3++;
    }
    while ( (unsigned int)(unsigned __int8)*v2 - 48 <= 9 );
  }
  v4 = sub_11FE300((__int64)a1, (char *)a1[7] + 1, v2);
  if ( v4 != (unsigned int)v4 )
    JUMPOUT(0x11FF0D0);
  *((_DWORD *)a1 + 26) = v4;
  return 506;
}
