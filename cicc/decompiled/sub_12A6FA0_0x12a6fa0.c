// Function: sub_12A6FA0
// Address: 0x12a6fa0
//
char *__fastcall sub_12A6FA0(__int64 a1)
{
  char v1; // al
  char *v2; // r8
  char *v3; // r8
  char v5; // al
  char v6; // al
  char v7; // al

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 15 )
    return "l";
  v2 = "f";
  if ( v1 != 2 )
  {
    v2 = "d";
    if ( v1 != 3 )
    {
      if ( !(unsigned __int8)sub_1642F90(a1, 8) )
      {
        v5 = sub_1642F90(a1, 16);
        v3 = (char *)&unk_3F7DA1E;
        if ( v5 )
          return v3;
        if ( !(unsigned __int8)sub_1642F90(a1, 32) )
        {
          v6 = sub_1642F90(a1, 64);
          v3 = "l";
          if ( !v6 )
          {
            v7 = sub_1642F90(a1, 128);
            v3 = "q";
            if ( !v7 )
              sub_127B630("unexpected LLVM type!", 0);
          }
          return v3;
        }
      }
      return "r";
    }
  }
  return v2;
}
