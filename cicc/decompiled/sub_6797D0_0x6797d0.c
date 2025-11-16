// Function: sub_6797D0
// Address: 0x6797d0
//
__int64 __fastcall sub_6797D0(unsigned __int16 a1, int a2)
{
  _WORD v3[176]; // [rsp+0h] [rbp-170h] BYREF
  int v4; // [rsp+160h] [rbp-10h]
  __int16 v5; // [rsp+164h] [rbp-Ch]

  memset(v3, 0, sizeof(v3));
  v3[37] = 257;
  *((_BYTE *)v3 + a1) = 1;
  v4 = 0;
  v5 = 0;
  return sub_7C64E0(0, v3, (unsigned int)(a2 + 4));
}
