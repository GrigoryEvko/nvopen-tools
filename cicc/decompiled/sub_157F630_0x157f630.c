// Function: sub_157F630
// Address: 0x157f630
//
char __fastcall sub_157F630(__int64 a1)
{
  unsigned __int64 v1; // rax
  int v2; // eax

  v1 = sub_157EBA0(a1);
  if ( !v1 )
    return 1;
  v2 = *(unsigned __int8 *)(v1 + 16);
  if ( (unsigned int)(v2 - 24) > 6 )
    return (unsigned int)(v2 - 32) > 2;
  else
    return (unsigned int)(v2 - 24) <= 4;
}
