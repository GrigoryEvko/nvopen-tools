// Function: sub_39824E0
// Address: 0x39824e0
//
__int64 __fastcall sub_39824E0(__int64 *a1, __int64 a2, unsigned __int16 a3)
{
  __int64 v3; // rax
  unsigned __int64 v5; // [rsp+8h] [rbp-8h] BYREF

  v3 = *a1;
  if ( a3 == 14 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 240) + 356LL) )
    {
      v5 = *(_QWORD *)(v3 + 8);
      return sub_39822D0((__int64)&v5, a2, 14);
    }
    else
    {
      v5 = *(unsigned int *)(v3 + 16);
      return sub_3982020(&v5, a2, 0xEu);
    }
  }
  else
  {
    v5 = *(unsigned int *)(v3 + 20);
    return sub_3982020(&v5, a2, a3);
  }
}
