// Function: sub_B2BAE0
// Address: 0xb2bae0
//
__int64 __fastcall sub_B2BAE0(__int64 a1)
{
  __int64 result; // rax
  _QWORD v2[4]; // [rsp-20h] [rbp-20h] BYREF

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 14 )
  {
    v2[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 120LL);
    if ( (unsigned __int8)sub_A74710(v2, *(_DWORD *)(a1 + 32) + 1, 81)
      || (unsigned __int8)sub_A74710(v2, *(_DWORD *)(a1 + 32) + 1, 83) )
    {
      return 1;
    }
    else
    {
      return sub_A74710(v2, *(_DWORD *)(a1 + 32) + 1, 84);
    }
  }
  return result;
}
