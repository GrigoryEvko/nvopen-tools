// Function: sub_33DD210
// Address: 0x33dd210
//
__int64 __fastcall sub_33DD210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]
  unsigned __int64 v9; // [rsp+10h] [rbp-20h]
  unsigned int v10; // [rsp+18h] [rbp-18h]

  sub_33DD090((__int64)&v7, a1, a2, a3, a5);
  if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    LOBYTE(a4) = (*(_QWORD *)a4 & ~v7) == 0;
  else
    LODWORD(a4) = sub_C446F0((__int64 *)a4, (__int64 *)&v7);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  return (unsigned int)a4;
}
