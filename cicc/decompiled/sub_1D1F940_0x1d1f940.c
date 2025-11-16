// Function: sub_1D1F940
// Address: 0x1d1f940
//
__int64 __fastcall sub_1D1F940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]
  __int64 v10; // [rsp+18h] [rbp-18h]

  v7 = 0;
  v8 = 1;
  v9 = 0;
  v10 = 1;
  sub_1D1F820(a1, a2, a3, (unsigned __int64 *)&v7, a5);
  if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    LOBYTE(a4) = (*(_QWORD *)a4 & ~v7) == 0;
  else
    LODWORD(a4) = sub_16A5A00((__int64 *)a4, &v7);
  if ( (unsigned int)v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( (unsigned int)v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  return (unsigned int)a4;
}
