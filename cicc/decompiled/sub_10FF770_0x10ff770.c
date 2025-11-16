// Function: sub_10FF770
// Address: 0x10ff770
//
__int64 __fastcall sub_10FF770(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, int a7, char a8)
{
  __int64 v10; // r12
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // esi
  _BYTE v19[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  if ( *(_QWORD *)(a3 + 8) == a4 )
    return a3;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[10] + 120LL))(a1[10]);
  if ( !v10 )
  {
    v20 = 257;
    v10 = sub_B51D30(a2, a3, a4, (__int64)v19, 0, 0);
    if ( (unsigned __int8)sub_920620(v10) )
    {
      if ( !a8 )
        a7 = *((_DWORD *)a1 + 26);
      if ( a6 || (a6 = a1[12]) != 0 )
        sub_B99FD0(v10, 3u, a6);
      sub_B45150(v10, a7);
    }
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a5,
      a1[7],
      a1[8]);
    v13 = *a1;
    v14 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v14 )
    {
      do
      {
        v15 = *(_QWORD *)(v13 + 8);
        v16 = *(_DWORD *)v13;
        v13 += 16;
        sub_B99FD0(v10, v16, v15);
      }
      while ( v14 != v13 );
    }
  }
  return v10;
}
