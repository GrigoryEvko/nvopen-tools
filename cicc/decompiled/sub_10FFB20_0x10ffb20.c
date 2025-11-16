// Function: sub_10FFB20
// Address: 0x10ffb20
//
__int64 __fastcall sub_10FFB20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r15
  __int64 v10; // r10
  int v11; // r13d
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdx
  unsigned int v15; // esi
  _BYTE v18[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 108) )
    return sub_B358C0(a1, 0x71u, a2, a3, a4, a5, a6, 0, 0);
  if ( a3 == *(_QWORD *)(a2 + 8) )
    return a2;
  v8 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80) + 120LL))(
         *(_QWORD *)(a1 + 80),
         45,
         a2,
         a3);
  if ( !v8 )
  {
    v19 = 257;
    v8 = sub_B51D30(45, a2, a3, (__int64)v18, 0, 0);
    if ( (unsigned __int8)sub_920620(v8) )
    {
      v10 = a6;
      v11 = a4;
      if ( !BYTE4(a4) )
        v11 = *(_DWORD *)(a1 + 104);
      if ( a6 || (v10 = *(_QWORD *)(a1 + 96)) != 0 )
        sub_B99FD0(v8, 3u, v10);
      sub_B45150(v8, v11);
    }
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v8,
      a5,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v12 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v14 = *(_QWORD *)(v12 + 8);
        v15 = *(_DWORD *)v12;
        v12 += 16;
        sub_B99FD0(v8, v15, v14);
      }
      while ( v13 != v12 );
    }
  }
  return v8;
}
