// Function: sub_2C13D90
// Address: 0x2c13d90
//
__int64 __fastcall sub_2C13D90(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // r12
  int v12; // r13d
  _BYTE v14[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  if ( *(_QWORD *)(a3 + 8) == a4 )
    return a3;
  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 80) + 120LL))(*(_QWORD *)(a1 + 80));
  if ( !v10 )
  {
    v15 = 257;
    v10 = sub_B51D30(a2, a3, a4, (__int64)v14, 0, 0);
    if ( (unsigned __int8)sub_920620(v10) )
    {
      v12 = *(_DWORD *)(a1 + 104);
      if ( BYTE4(a7) )
        v12 = a7;
      if ( a6 || (a6 = *(_QWORD *)(a1 + 96)) != 0 )
        sub_B99FD0(v10, 3u, a6);
      sub_B45150(v10, v12);
    }
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v10,
      a5,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    sub_94AAF0((unsigned int **)a1, v10);
  }
  return v10;
}
