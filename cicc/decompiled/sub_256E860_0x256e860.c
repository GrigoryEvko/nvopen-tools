// Function: sub_256E860
// Address: 0x256e860
//
__int64 __fastcall sub_256E860(unsigned __int8 ***a1, __int64 a2)
{
  unsigned __int8 *v2; // rdx
  unsigned __int8 **v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  unsigned __int8 *v9[4]; // [rsp+0h] [rbp-20h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v2 = **(unsigned __int8 ***)(a2 - 8);
    if ( (unsigned int)*v2 - 12 <= 1 )
      return 1;
  }
  else
  {
    v2 = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( (unsigned int)*v2 - 12 <= 1 )
      return 1;
  }
  if ( v2 != **a1 )
  {
    v4 = a1[1];
    v9[0] = **a1;
    v9[1] = (unsigned __int8 *)a2;
    if ( sub_250C1E0(v9, (__int64)v4[26]) )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v8 = *(_QWORD *)(a2 - 8);
      else
        v8 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      if ( (unsigned __int8)sub_256E5A0((__int64)a1[1], v8, **a1, v5, v6, v7) )
        *(_DWORD *)a1[2] = 0;
    }
  }
  return 1;
}
