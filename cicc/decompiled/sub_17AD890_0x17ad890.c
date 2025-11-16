// Function: sub_17AD890
// Address: 0x17ad890
//
__int64 __fastcall sub_17AD890(
        __int64 *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // eax
  unsigned int v11; // ebx
  unsigned __int64 v12; // r13
  unsigned int v13; // r15d
  __int64 v14; // rbx
  __int64 v15; // r14
  _QWORD *v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  unsigned __int64 v20; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+8h] [rbp-68h]
  unsigned __int64 v22; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+38h] [rbp-38h]

  v10 = sub_16431D0(*a2);
  v25 = v10;
  v11 = v10;
  if ( v10 > 0x40 )
  {
    sub_16A4EF0((__int64)&v24, 0, 0);
    v27 = v11;
    sub_16A4EF0((__int64)&v26, 0, 0);
    v21 = v11;
    sub_16A4EF0((__int64)&v20, -1, 1);
    v23 = v21;
    if ( v21 > 0x40 )
    {
      sub_16A4FD0((__int64)&v22, (const void **)&v20);
      goto LABEL_4;
    }
  }
  else
  {
    v27 = v10;
    v21 = v10;
    v24 = 0;
    v26 = 0;
    v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
    v23 = v10;
  }
  v22 = v20;
LABEL_4:
  v12 = sub_17A9010((__int64)a1, (__int64)a2, (__int64)&v22, (__int64)&v24, 0, (__int64)a2);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v12 )
  {
    v13 = 1;
    if ( a2 != (__int64 *)v12 )
    {
      v14 = a2[1];
      if ( v14 )
      {
        v15 = *a1;
        do
        {
          v16 = sub_1648700(v14);
          sub_170B990(v15, (__int64)v16);
          v14 = *(_QWORD *)(v14 + 8);
        }
        while ( v14 );
        v13 = 1;
        sub_164D160((__int64)a2, v12, a3, a4, a5, a6, v17, v18, a9, a10);
      }
    }
  }
  else
  {
    v13 = 0;
  }
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  return v13;
}
