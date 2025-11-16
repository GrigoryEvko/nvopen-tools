// Function: sub_E0D090
// Address: 0xe0d090
//
unsigned __int64 __fastcall sub_E0D090(_QWORD *a1, unsigned __int64 *a2)
{
  char *v3; // r9
  int v5; // ecx
  char *v6; // rsi
  char *v7; // r9
  unsigned __int64 v8; // rdi
  __int64 v9; // rcx
  unsigned __int64 result; // rax

  v3 = (char *)a1[1];
  v5 = *v3;
  v6 = v3 + 1;
  v7 = &v3[*a1];
  v8 = 0;
  v9 = v5 - 48;
  while ( 1 )
  {
    a1[1] = v6;
    v8 = v9 + 10 * v8;
    result = v7 - v6;
    *a1 = v7 - v6;
    if ( v6 == v7 )
      break;
    LODWORD(v9) = *v6 - 48;
    if ( (unsigned int)v9 > 9 )
    {
      *a2 = v8;
      return result;
    }
    v9 = (int)v9;
    ++v6;
    result = 0xCCCCCCCCCCCCCCCDLL * (0xFFFFFFFFLL - (int)v9);
    if ( (0xFFFFFFFFLL - (int)v9) / 0xAuLL < v8 )
    {
      *a1 = 0;
      a1[1] = 0;
      return result;
    }
  }
  a1[1] = 0;
  return result;
}
