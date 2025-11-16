// Function: sub_1BF9660
// Address: 0x1bf9660
//
unsigned __int64 __fastcall sub_1BF9660(__int64 a1)
{
  unsigned int v2; // ebx
  bool v3; // zf
  unsigned int v4; // eax
  unsigned int v5; // [rsp+4h] [rbp-1Ch] BYREF
  unsigned int v6; // [rsp+8h] [rbp-18h] BYREF
  char v7; // [rsp+Ch] [rbp-14h]

  sub_1C2ECF0(&v6, a1);
  if ( !v7 )
    return 0;
  v2 = v6;
  v3 = (unsigned __int8)sub_1C2EF70(a1, &v5) == 0;
  v4 = 0;
  if ( !v3 )
    v4 = v5;
  if ( v2 <= 0x400 )
    return ((unsigned __int64)v4 << 32) | (v2 + 31) & 0xFFFFFFE0;
  else
    return 0;
}
