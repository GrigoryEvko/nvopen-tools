// Function: sub_740900
// Address: 0x740900
//
__m128i *__fastcall sub_740900(const __m128i *a1, int a2)
{
  __m128i *result; // rax
  __m128i *v3; // [rsp+8h] [rbp-28h]
  int v4[3]; // [rsp+1Ch] [rbp-14h] BYREF

  v4[0] = 0;
  if ( !a2 || dword_4F07270[0] == unk_4F073B8 )
    return sub_740630(a1);
  sub_7296C0(v4);
  result = sub_740630(a1);
  if ( v4[0] )
  {
    v3 = result;
    sub_729730(v4[0]);
    return v3;
  }
  return result;
}
