// Function: sub_3401900
// Address: 0x3401900
//
unsigned __int8 *__fastcall sub_3401900(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7)
{
  unsigned int v10; // edx
  unsigned __int64 v11; // r8
  bool v12; // al
  void *v13; // r8
  unsigned __int8 *v14; // r12
  int v15; // r9d
  unsigned __int64 v17; // [rsp+8h] [rbp-88h]
  int v18; // [rsp+10h] [rbp-80h]
  unsigned int v19; // [rsp+10h] [rbp-80h]
  bool v20; // [rsp+10h] [rbp-80h]
  void *v21; // [rsp+10h] [rbp-80h]
  const void **v23; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-68h]
  unsigned __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-58h]
  const void **v27; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-48h]
  const void **v29; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-38h]

  if ( *(_DWORD *)(a5 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)a5 )
      goto LABEL_3;
    return sub_3400BD0(a1, 0, a2, a3, a4, 0, a7, 0);
  }
  v18 = *(_DWORD *)(a5 + 8);
  if ( v18 - (unsigned int)sub_C444A0(a5) <= 0x40 && !**(_QWORD **)a5 )
    return sub_3400BD0(a1, 0, a2, a3, a4, 0, a7, 0);
LABEL_3:
  if ( !a6 )
  {
LABEL_32:
    sub_34007B0(a1, a5, a2, a3, a4, 0, a7, 0);
    return sub_33FAF80(a1, 373, a2, a3, a4, v15, a7);
  }
  sub_988CD0((__int64)&v27, **(_QWORD **)(a1 + 40), 0x40u);
  v24 = v28;
  if ( v28 > 0x40 )
    sub_C43780((__int64)&v23, (const void **)&v27);
  else
    v23 = v27;
  sub_C46A40((__int64)&v23, 1);
  v10 = v24;
  v11 = (unsigned __int64)v23;
  v24 = 0;
  v26 = v10;
  v25 = (unsigned __int64)v23;
  if ( v30 <= 0x40 )
  {
    v12 = v29 == v23;
  }
  else
  {
    v17 = (unsigned __int64)v23;
    v19 = v10;
    v12 = sub_C43C50((__int64)&v29, (const void **)&v25);
    v11 = v17;
    v10 = v19;
  }
  if ( v10 > 0x40 )
  {
    if ( v11 )
    {
      v20 = v12;
      j_j___libc_free_0_0(v11);
      v12 = v20;
      if ( v24 > 0x40 )
      {
        if ( v23 )
        {
          j_j___libc_free_0_0((unsigned __int64)v23);
          v12 = v20;
        }
      }
    }
  }
  if ( !v12 )
  {
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0((unsigned __int64)v29);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0((unsigned __int64)v27);
    goto LABEL_32;
  }
  v13 = v27;
  if ( v28 > 0x40 )
    v13 = (void *)*v27;
  v24 = *(_DWORD *)(a5 + 8);
  if ( v24 > 0x40 )
  {
    v21 = v13;
    sub_C43780((__int64)&v23, (const void **)a5);
    v13 = v21;
  }
  else
  {
    v23 = *(const void ***)a5;
  }
  sub_C47170((__int64)&v23, (unsigned __int64)v13);
  v26 = v24;
  v24 = 0;
  v25 = (unsigned __int64)v23;
  v14 = sub_34007B0(a1, (__int64)&v25, a2, a3, a4, 0, a7, 0);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0((unsigned __int64)v23);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0((unsigned __int64)v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0((unsigned __int64)v27);
  return v14;
}
