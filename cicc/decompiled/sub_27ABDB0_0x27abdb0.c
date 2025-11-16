// Function: sub_27ABDB0
// Address: 0x27abdb0
//
__int64 __fastcall sub_27ABDB0(char *a1)
{
  char v1; // al
  unsigned int v3; // r8d
  bool v4; // al
  int v5; // eax

  v1 = *a1;
  if ( (unsigned __int8)(*a1 - 61) <= 1u )
    return 1;
  if ( v1 != 34 )
    goto LABEL_4;
  v4 = sub_B49E00((__int64)a1);
  v3 = 1;
  if ( v4 )
  {
    v1 = *a1;
LABEL_4:
    v3 = 0;
    if ( v1 == 85 )
    {
      LOBYTE(v5) = sub_B49E00((__int64)a1);
      return v5 ^ 1u;
    }
  }
  return v3;
}
