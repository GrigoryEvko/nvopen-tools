// Function: sub_D5D560
// Address: 0xd5d560
//
__int64 __fastcall sub_D5D560(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  unsigned int v5[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = sub_D5BAA0((unsigned __int8 *)a1);
  if ( v2 )
  {
    if ( a2 )
    {
      v3 = v2;
      if ( sub_981210(*a2, v2, v5)
        && (a2[((unsigned __int64)v5[0] >> 6) + 1] & (1LL << SLOBYTE(v5[0]))) == 0
        && (((int)*(unsigned __int8 *)(*a2 + (v5[0] >> 2)) >> (2 * (v5[0] & 3))) & 3) != 0
        && sub_D5D4E0(v3, v5[0]) )
      {
        return *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      }
    }
  }
  if ( (sub_D5BB80((unsigned __int8 *)a1) & 4) != 0 )
    return sub_B494D0(a1, 2);
  return 0;
}
