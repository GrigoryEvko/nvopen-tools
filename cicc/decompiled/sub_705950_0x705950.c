// Function: sub_705950
// Address: 0x705950
//
__int64 __fastcall sub_705950(unsigned __int16 a1, const char *a2)
{
  size_t v2; // rbx
  size_t v4; // rbx
  __int16 v5; // [rsp+0h] [rbp-60h] BYREF
  _BYTE dest[94]; // [rsp+2h] [rbp-5Eh] BYREF

  if ( (*a2 == 95 || unk_4D044BC) && (sub_885C00(a1, a2), *a2 == 95) )
  {
    v4 = strlen(a2);
    memcpy(&v5, a2, v4 + 1);
    dest[v4 - 1] = 95;
    dest[v4 - 2] = 95;
    dest[v4] = 0;
  }
  else
  {
    v2 = strlen(a2);
    v5 = 24415;
    memcpy(dest, a2, v2 + 1);
    sub_885C00(a1, &v5);
    dest[v2 + 1] = 95;
    dest[v2] = 95;
    dest[v2 + 2] = 0;
  }
  return sub_885C00(a1, &v5);
}
