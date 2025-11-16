// Function: sub_9B01D0
// Address: 0x9b01d0
//
__int64 __fastcall sub_9B01D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i *v7; // r8
  unsigned int v8; // r15d
  unsigned int v9; // eax
  unsigned __int64 v10; // rdi
  __m128i *v12; // [rsp+0h] [rbp-90h]
  unsigned __int64 v13; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-78h]
  __int64 v15; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-68h]
  __int64 v17; // [rsp+30h] [rbp-60h]
  unsigned int v18; // [rsp+38h] [rbp-58h]
  __int64 v19; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+48h] [rbp-48h]
  __int64 v21; // [rsp+50h] [rbp-40h]
  unsigned int v22; // [rsp+58h] [rbp-38h]

  v7 = *(__m128i **)(a2 + 8);
  v8 = *(_DWORD *)a2 + 1;
  v9 = *(_DWORD *)(a4 + 8);
  v14 = v9;
  if ( v9 > 0x40 )
  {
    v12 = v7;
    sub_C43780(&v13, a4);
    v9 = v14;
    v7 = v12;
    if ( v14 > 0x40 )
    {
      sub_C47690(&v13, 1);
      v7 = v12;
      goto LABEL_6;
    }
  }
  else
  {
    v13 = *(_QWORD *)a4;
  }
  v10 = 0;
  if ( v9 >= 2 )
    v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & (2 * v13);
  v13 = v10;
LABEL_6:
  sub_9B0110((__int64)&v19, a3, (__int64)&v13, v8, v7);
  sub_9B0110((__int64)&v15, a3, a4, *(_DWORD *)a2 + 1, *(__m128i **)(a2 + 8));
  (*(void (__fastcall **)(__int64, _QWORD, __int64 *, __int64 *))(a2 + 16))(a1, *(_QWORD *)(a2 + 24), &v15, &v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return a1;
}
