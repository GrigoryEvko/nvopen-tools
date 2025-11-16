// Function: sub_1BF5810
// Address: 0x1bf5810
//
__int64 __fastcall sub_1BF5810(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  int v4; // edx
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v6 = a3;
  v4 = *(_DWORD *)(a1 + 40);
  if ( !v4 || ((v4 != 1) & (a4 ^ 1)) != 0 )
  {
    sub_1BF5800(a1);
    return 0;
  }
  else if ( *(_DWORD *)(a1 + 56) == 1 )
  {
    sub_1BF4B30(*(__int64 **)(a1 + 80), a1, &v6);
    return 0;
  }
  else
  {
    return 1;
  }
}
