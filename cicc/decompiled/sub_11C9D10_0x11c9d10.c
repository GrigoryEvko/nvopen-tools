// Function: sub_11C9D10
// Address: 0x11c9d10
//
char __fastcall sub_11C9D10(__int64 *a1, __int64 *a2, _BYTE *a3, size_t a4)
{
  char result; // al
  unsigned int v5[5]; // [rsp+Ch] [rbp-14h] BYREF

  result = sub_980AF0(*a2, a3, a4, v5);
  if ( result )
    return sub_11C99B0(a1, a2, v5[0]);
  return result;
}
