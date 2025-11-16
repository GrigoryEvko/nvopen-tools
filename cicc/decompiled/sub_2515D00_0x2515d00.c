// Function: sub_2515D00
// Address: 0x2515d00
//
void __fastcall sub_2515D00(__int64 a1, __m128i *a2, int *a3, __int64 a4, __int64 a5, char a6)
{
  __int64 *v6; // r14
  __int64 *v10; // rdi
  __int64 *v11; // r14
  int *v12; // r13
  int v13; // edx
  __int64 *v15; // [rsp+10h] [rbp-A0h]
  __int64 v17; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v18; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-78h]
  char v20; // [rsp+40h] [rbp-70h] BYREF

  v6 = (__int64 *)a2;
  v17 = a5;
  sub_250D360((__int64)&v18, a2);
  v10 = v18;
  v15 = &v18[2 * v19];
  if ( v15 != v18 )
  {
    v11 = v18;
    do
    {
      sub_25157B0(
        a1,
        v11,
        (__int64)a3,
        a4,
        (unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD *, __int64 **))sub_2506DA0,
        (__int64)&v17);
      if ( a6 )
        break;
      v11 += 2;
    }
    while ( v15 != v11 );
    v6 = (__int64 *)a2;
    v10 = v18;
  }
  if ( v10 != (__int64 *)&v20 )
    _libc_free((unsigned __int64)v10);
  v12 = &a3[a4];
  while ( v12 != a3 )
  {
    v13 = *a3++;
    sub_2512BA0(a1, v6, v13, a5);
  }
}
