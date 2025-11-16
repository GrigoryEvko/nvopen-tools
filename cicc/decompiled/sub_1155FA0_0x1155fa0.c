// Function: sub_1155FA0
// Address: 0x1155fa0
//
__int64 __fastcall sub_1155FA0(__int64 *a1, int a2, __int64 a3, __int64 a4, int a5, char a6, __int64 a7, __int64 a8)
{
  __int64 v10; // r12
  int v12; // ebx
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // esi
  _BYTE v19[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[10] + 16LL))(a1[10]);
  if ( !v10 )
  {
    v20 = 257;
    v10 = sub_B504D0(a2, a3, a4, (__int64)v19, 0, 0);
    if ( (unsigned __int8)sub_920620(v10) )
    {
      v12 = a5;
      if ( !a6 )
        v12 = *((_DWORD *)a1 + 26);
      if ( a8 || (a8 = a1[12]) != 0 )
        sub_B99FD0(v10, 3u, a8);
      sub_B45150(v10, v12);
    }
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a7,
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
