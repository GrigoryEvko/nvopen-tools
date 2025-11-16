// Function: sub_3006DC0
// Address: 0x3006dc0
//
__int64 __fastcall sub_3006DC0(int a1, int a2, char a3, char a4, char a5, char a6)
{
  char v6; // al
  char v7; // r10
  char v8; // r11
  bool v9; // bl
  bool v11; // si

  v9 = a2 == 7;
  if ( a1 == 56 && a2 == 7 )
    return 234;
  v11 = a2 == 8;
  if ( a1 == 64 && v11 )
    return 235;
  if ( v6 && a4 )
    return 236;
  if ( a3 && v7 )
    return 237;
  if ( a5 && a1 == 64 )
    return 238;
  if ( a1 == 80 && a6 )
    return 239;
  if ( v8 && a1 == 96 )
    return 240;
  if ( a1 == 112 && v9 )
    return 241;
  if ( v11 && a1 == 128 )
    return 242;
  if ( v6 && a1 == 64 )
    return 243;
  if ( a3 && a1 == 96 )
    return 244;
  if ( a5 && a1 == 128 )
    return 245;
  if ( a1 == 160 && a6 )
    return 246;
  if ( v8 && a1 == 192 )
    return 247;
  if ( a1 == 224 && v9 )
    return 248;
  if ( v11 && a1 == 256 )
    return 249;
  if ( v6 && a1 == 128 )
    return 250;
  if ( a3 && a1 == 192 )
    return 251;
  if ( a5 && a1 == 256 )
    return 252;
  if ( a1 == 320 && a6 )
    return 253;
  if ( v8 && a1 == 384 )
    return 254;
  if ( a1 == 448 && v9 )
    return 255;
  if ( v11 && a1 == 512 )
    return 256;
  if ( v6 && a1 == 256 )
    return 257;
  if ( a3 && a1 == 384 )
    return 258;
  if ( a5 && a1 == 512 )
    return 259;
  if ( !v6 || a1 != 512 )
    BUG();
  return 260;
}
