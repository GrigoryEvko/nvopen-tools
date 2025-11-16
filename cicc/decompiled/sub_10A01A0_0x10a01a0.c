// Function: sub_10A01A0
// Address: 0x10a01a0
//
__int64 __fastcall sub_10A01A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rdx
  unsigned int v11; // esi
  _BYTE v12[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v13; // [rsp+20h] [rbp-40h]

  v5 = sub_AD8D80(*(_QWORD *)(a2 + 8), a3);
  v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 16LL))(a1[10], 29, a2, v5);
  if ( !v6 )
  {
    v13 = 257;
    v6 = sub_B504D0(29, a2, v5, (__int64)v12, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v6,
      a4,
      a1[7],
      a1[8]);
    v8 = *a1;
    v9 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v9 )
    {
      do
      {
        v10 = *(_QWORD *)(v8 + 8);
        v11 = *(_DWORD *)v8;
        v8 += 16;
        sub_B99FD0(v6, v11, v10);
      }
      while ( v9 != v8 );
    }
  }
  return v6;
}
