// Function: sub_388B0A0
// Address: 0x388b0a0
//
__int64 __fastcall sub_388B0A0(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rsi
  const char *v4; // [rsp+0h] [rbp-30h] BYREF
  char v5; // [rsp+10h] [rbp-20h]
  char v6; // [rsp+11h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 64) == 377 )
  {
    sub_2240AE0(a2, (unsigned __int64 *)(a1 + 72));
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 56);
    v6 = 1;
    v5 = 3;
    v4 = "expected string constant";
    return sub_38814C0(a1 + 8, v2, (__int64)&v4);
  }
}
