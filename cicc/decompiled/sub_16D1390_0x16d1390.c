// Function: sub_16D1390
// Address: 0x16d1390
//
_QWORD *__fastcall sub_16D1390(
        __int64 *a1,
        _QWORD *a2,
        unsigned __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  unsigned __int8 v10; // bl
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r12
  __int64 v13; // rdi
  _QWORD *result; // rax
  _BYTE v15[16]; // [rsp+0h] [rbp-190h] BYREF
  _QWORD *v16; // [rsp+10h] [rbp-180h]
  _QWORD v17[4]; // [rsp+20h] [rbp-170h] BYREF
  __int64 *v18; // [rsp+40h] [rbp-150h]
  __int64 v19; // [rsp+50h] [rbp-140h] BYREF
  __int64 *v20; // [rsp+60h] [rbp-130h]
  __int64 v21; // [rsp+70h] [rbp-120h] BYREF
  __int64 v22; // [rsp+80h] [rbp-110h]
  __int64 v23; // [rsp+90h] [rbp-100h]
  unsigned __int64 v24; // [rsp+98h] [rbp-F8h]
  unsigned int v25; // [rsp+A0h] [rbp-F0h]
  char v26; // [rsp+A8h] [rbp-E8h] BYREF

  v10 = a6;
  sub_16D0E30((__int64)v15, a1, a3, a4, a5, a6, a7, a8, a9, a10);
  sub_16CFD20(a1, a2, (__int64)v15, v10);
  v11 = v24;
  v12 = v24 + 48LL * v25;
  if ( v24 != v12 )
  {
    do
    {
      v12 -= 48LL;
      v13 = *(_QWORD *)(v12 + 16);
      if ( v13 != v12 + 32 )
        j_j___libc_free_0(v13, *(_QWORD *)(v12 + 32) + 1LL);
    }
    while ( v11 != v12 );
    v12 = v24;
  }
  if ( (char *)v12 != &v26 )
    _libc_free(v12);
  if ( v22 )
    j_j___libc_free_0(v22, v23 - v22);
  if ( v20 != &v21 )
    j_j___libc_free_0(v20, v21 + 1);
  if ( v18 != &v19 )
    j_j___libc_free_0(v18, v19 + 1);
  result = v17;
  if ( v16 != v17 )
    return (_QWORD *)j_j___libc_free_0(v16, v17[0] + 1LL);
  return result;
}
