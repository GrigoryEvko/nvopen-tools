// Function: sub_AF29C0
// Address: 0xaf29c0
//
unsigned __int64 __fastcall sub_AF29C0(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned __int64 *v2; // rdi
  unsigned __int64 v3; // rax
  int v4; // ecx

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(unsigned __int64 **)(a1 - 32);
  else
    v2 = (unsigned __int64 *)(a1 - 16 - 8LL * ((v1 >> 2) & 0xF));
  v3 = *v2;
  if ( !*v2 )
    return 0;
  v4 = *(unsigned __int8 *)v3;
  if ( (unsigned int)(v4 - 25) <= 1 )
    return v3 & 0xFFFFFFFFFFFFFFFBLL;
  if ( (_BYTE)v4 == 7 )
    return v3 | 4;
  else
    return 0;
}
