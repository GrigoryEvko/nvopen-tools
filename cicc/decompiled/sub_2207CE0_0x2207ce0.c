// Function: sub_2207CE0
// Address: 0x2207ce0
//
__int64 __fastcall sub_2207CE0(__int64 a1, const char *a2, char a3)
{
  char *v3; // rax
  const char *v4; // rbp
  FILE *v5; // rax

  v3 = sub_2207BE0(a3);
  if ( !v3 )
    return 0;
  v4 = v3;
  if ( sub_2207CD0((_QWORD *)a1) )
    return 0;
  v5 = fopen64(a2, v4);
  *(_QWORD *)a1 = v5;
  if ( !v5 )
    return 0;
  *(_BYTE *)(a1 + 8) = 1;
  return a1;
}
