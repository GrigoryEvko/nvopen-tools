// Function: sub_21680A0
// Address: 0x21680a0
//
_QWORD *__fastcall sub_21680A0(
        int a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int *a9,
        int a10)
{
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int128 v16; // [rsp-30h] [rbp-90h]
  int v18; // [rsp+20h] [rbp-40h] BYREF
  int v20; // [rsp+28h] [rbp-38h] BYREF

  if ( *(_BYTE *)(a8 + 4) )
    v18 = *(_DWORD *)a8;
  if ( *((_BYTE *)a9 + 4) )
    v20 = *a9;
  v13 = (_QWORD *)sub_22077B0(83368);
  v14 = v13;
  if ( v13 )
  {
    *((_QWORD *)&v16 + 1) = a6;
    *(_QWORD *)&v16 = a5;
    sub_2168030(v13, a1, a2, a3, a4, a7, v16, &v18, (__int64)&v20, a10);
  }
  return v14;
}
