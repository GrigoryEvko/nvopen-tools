// Function: sub_7BE200
// Address: 0x7be200
//
__int64 __fastcall sub_7BE200(unsigned int a1, unsigned int a2, __int64 *a3)
{
  FILE *v5; // rsi
  _DWORD *v6; // r12
  __int64 v8; // [rsp+8h] [rbp-38h] BYREF
  __m128i v9[3]; // [rsp+10h] [rbp-30h] BYREF

  v5 = (FILE *)dword_4F07508;
  v6 = sub_67D9D0(a1, dword_4F07508);
  v8 = *a3;
  if ( (_DWORD)v8 )
  {
    v9[0] = 0u;
    sub_6855B0(a2, (FILE *)&v8, v9);
    v5 = (FILE *)v9;
    sub_67E370((__int64)v6, v9);
  }
  return sub_685910((__int64)v6, v5);
}
