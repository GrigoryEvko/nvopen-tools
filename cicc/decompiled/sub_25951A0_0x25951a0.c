// Function: sub_25951A0
// Address: 0x25951a0
//
__int64 __fastcall sub_25951A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  char v3; // r8
  __int64 result; // rax
  char v5; // [rsp+Fh] [rbp-31h] BYREF
  __m128i v6[3]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_25096F0((_QWORD *)(a1 + 72));
  sub_250D230((unsigned __int64 *)v6, v2, 4, 0);
  v3 = sub_25950E0(a2, a1, v6, 0, &v5, 0, 0);
  result = 1;
  if ( !v3 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  return result;
}
