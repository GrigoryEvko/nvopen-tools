// Function: sub_3885DD0
// Address: 0x3885dd0
//
__int64 __fastcall sub_3885DD0(unsigned __int8 **a1)
{
  __int64 v1; // rax
  char *v2; // r8
  char *v3; // rax
  unsigned __int64 v4; // r13

  v1 = (__int64)*a1;
  if ( (unsigned int)**a1 - 48 > 9 )
    JUMPOUT(0x3885DB0);
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
  v4 = sub_3881F70((__int64)a1, (char *)a1[6] + 1, v2);
  if ( v4 != (unsigned int)v4 )
    JUMPOUT(0x3885D80);
  *((_DWORD *)a1 + 24) = v4;
  return 370;
}
