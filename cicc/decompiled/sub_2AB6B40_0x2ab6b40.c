// Function: sub_2AB6B40
// Address: 0x2ab6b40
//
__int64 __fastcall sub_2AB6B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx

  v4 = a3;
  if ( (unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 72) + 512LL, a2, a3, a4)
    || v4 && (unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 72) + 672LL, a2, v5, v6) )
  {
    return 1;
  }
  else
  {
    return sub_B19060(a1 + 80, a2, v5, v6);
  }
}
