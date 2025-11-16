// Function: sub_16F78D0
// Address: 0x16f78d0
//
__int64 __fastcall sub_16F78D0(__int64 a1, unsigned int a2)
{
  char *v2; // rdx

  if ( a2 > 0x7F )
    goto LABEL_8;
  v2 = *(char **)(a1 + 40);
  if ( v2 == *(char **)(a1 + 48) )
    return 0;
  if ( *v2 < 0 )
LABEL_8:
    sub_16BD130("Not dealing with this yet", 1u);
  if ( *v2 != a2 )
    return 0;
  ++*(_DWORD *)(a1 + 60);
  *(_QWORD *)(a1 + 40) = v2 + 1;
  return 1;
}
