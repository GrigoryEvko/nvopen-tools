// Function: sub_3261DF0
// Address: 0x3261df0
//
__int64 __fastcall sub_3261DF0(int a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6)
{
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]

  v9 = a3;
  v10 = a4;
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 17) > 0xD3u
      || !a6
      || *(_QWORD *)(a2 + 8LL * (unsigned __int16)a3 + 112) && !*(_BYTE *)(a2 + 500LL * (unsigned __int16)a3 + 6570) )
    {
      return sub_3400BD0(a5, 0, a1, v9, v10, 0, 0);
    }
  }
  else if ( !sub_30070B0((__int64)&v9) || !a6 )
  {
    return sub_3400BD0(a5, 0, a1, v9, v10, 0, 0);
  }
  return 0;
}
