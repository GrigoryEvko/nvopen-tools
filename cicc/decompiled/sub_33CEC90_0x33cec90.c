// Function: sub_33CEC90
// Address: 0x33cec90
//
__int64 __fastcall sub_33CEC90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 v9; // r8
  unsigned __int8 *v10; // rsi
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int8 *v15; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v14 = v5;
  if ( v5 )
  {
    sub_B96E90((__int64)&v14, v5, 1);
    v6 = v14;
    if ( v14 )
    {
      if ( *(_DWORD *)(a1 + 72) || *(_QWORD *)a3 == v14 )
      {
        v7 = *(_DWORD *)(a3 + 8);
        if ( *(_DWORD *)(a2 + 72) <= v7 )
          v7 = *(_DWORD *)(a2 + 72);
        *(_DWORD *)(a2 + 72) = v7;
      }
      else
      {
        v9 = *(_QWORD *)(a2 + 80);
        v15 = 0;
        if ( v9 )
        {
          sub_B91220(a2 + 80, v9);
          v10 = v15;
          *(_QWORD *)(a2 + 80) = v15;
          if ( v10 )
            sub_B976B0((__int64)&v15, v10, a2 + 80);
          v6 = v14;
          v11 = *(_DWORD *)(a3 + 8);
          if ( *(_DWORD *)(a2 + 72) <= v11 )
            v11 = *(_DWORD *)(a2 + 72);
          *(_DWORD *)(a2 + 72) = v11;
          if ( !v6 )
            return a2;
        }
        else
        {
          v13 = *(_DWORD *)(a2 + 72);
          if ( *(_DWORD *)(a3 + 8) <= v13 )
            v13 = *(_DWORD *)(a3 + 8);
          *(_DWORD *)(a2 + 72) = v13;
        }
      }
      sub_B91220((__int64)&v14, v6);
      return a2;
    }
  }
  v12 = *(_DWORD *)(a3 + 8);
  if ( *(_DWORD *)(a2 + 72) <= v12 )
    v12 = *(_DWORD *)(a2 + 72);
  *(_DWORD *)(a2 + 72) = v12;
  return a2;
}
