// Function: sub_2C137C0
// Address: 0x2c137c0
//
__int64 __fastcall sub_2C137C0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // r12
  int v11; // r14d
  _BYTE v14[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 80) + 16LL))(*(_QWORD *)(a1 + 80));
  if ( !v9 )
  {
    v15 = 257;
    v9 = sub_B504D0(a2, a3, a4, (__int64)v14, 0, 0);
    if ( (unsigned __int8)sub_920620(v9) )
    {
      v11 = *(_DWORD *)(a1 + 104);
      if ( BYTE4(a5) )
        v11 = a5;
      if ( a7 || (a7 = *(_QWORD *)(a1 + 96)) != 0 )
        sub_B99FD0(v9, 3u, a7);
      sub_B45150(v9, v11);
    }
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v9,
      a6,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    sub_94AAF0((unsigned int **)a1, v9);
  }
  return v9;
}
