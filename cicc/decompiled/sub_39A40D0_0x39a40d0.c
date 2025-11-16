// Function: sub_39A40D0
// Address: 0x39a40d0
//
void __fastcall sub_39A40D0(__int64 *a1, __int64 a2, _BYTE *a3, size_t a4)
{
  void *v4; // r12
  size_t v5; // rbx
  unsigned __int16 v6; // ax

  if ( a4 )
  {
    v4 = a3;
    v5 = a4;
    if ( *a3 == 1 )
    {
      v5 = a4 - 1;
      v4 = a3 + 1;
    }
    v6 = sub_398C0A0(a1[25]);
    sub_39A3F30(a1, a2, v6 < 4u ? 8199 : 110, v4, v5);
  }
}
