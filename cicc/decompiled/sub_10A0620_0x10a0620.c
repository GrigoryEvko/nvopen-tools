// Function: sub_10A0620
// Address: 0x10a0620
//
__int64 __fastcall sub_10A0620(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v8; // rdx
  int v9; // r14d
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // esi
  _BYTE v14[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v15; // [rsp+20h] [rbp-40h]

  if ( a3 == *(_QWORD *)(a2 + 8) )
    return a2;
  v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(a1[10], 40, a2, a3);
  if ( !v6 )
  {
    v15 = 257;
    v6 = sub_B51D30(40, a2, a3, (__int64)v14, 0, 0);
    if ( (unsigned __int8)sub_920620(v6) )
    {
      v8 = a1[12];
      v9 = *((_DWORD *)a1 + 26);
      if ( v8 )
        sub_B99FD0(v6, 3u, v8);
      sub_B45150(v6, v9);
    }
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v6,
      a4,
      a1[7],
      a1[8]);
    v10 = *a1;
    v11 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v11 )
    {
      do
      {
        v12 = *(_QWORD *)(v10 + 8);
        v13 = *(_DWORD *)v10;
        v10 += 16;
        sub_B99FD0(v6, v13, v12);
      }
      while ( v11 != v10 );
    }
  }
  return v6;
}
