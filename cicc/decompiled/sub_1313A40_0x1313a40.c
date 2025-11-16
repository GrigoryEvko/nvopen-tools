// Function: sub_1313A40
// Address: 0x1313a40
//
void __fastcall sub_1313A40(_BYTE *a1)
{
  char *v1; // rbx
  char v2; // al
  char v3; // al
  char v4; // t0

  v1 = a1 + 816;
  do
  {
    v2 = a1[816];
    if ( (unsigned __int8)v2 <= 2u )
    {
      v2 = 1;
      if ( !unk_4C6F030 && *a1 && (char)a1[1] <= 0 )
        v2 = sub_1313A30();
    }
    v4 = v2;
    v3 = *v1;
    *v1 = v4;
  }
  while ( v3 == 2 );
  sub_1313270((__int64)a1);
}
