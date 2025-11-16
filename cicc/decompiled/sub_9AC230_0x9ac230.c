// Function: sub_9AC230
// Address: 0x9ac230
//
__int64 __fastcall sub_9AC230(__int64 a1, __int64 a2, __m128i *a3, unsigned int a4)
{
  unsigned int v6; // r12d
  unsigned int v7; // ebx
  unsigned __int64 v9; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-48h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-38h]

  v6 = a2;
  v7 = *(_DWORD *)(a2 + 8);
  v10 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43690(&v9, 0, 0);
    v12 = v7;
    sub_C43690(&v11, 0, 0);
  }
  else
  {
    v9 = 0;
    v12 = v7;
    v11 = 0;
  }
  sub_9AC0E0(a1, &v9, a4, a3);
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
    LOBYTE(v6) = (*(_QWORD *)a2 & ~v9) == 0;
  else
    v6 = sub_C446F0(a2, &v9);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  return v6;
}
