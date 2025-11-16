// Function: sub_2222860
// Address: 0x2222860
//
__int64 __fastcall sub_2222860(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        char a6,
        __int128 a7,
        __int64 a8)
{
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v17[2]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v18[64]; // [rsp+30h] [rbp-40h] BYREF

  if ( !a8 )
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a1 + 16LL))(
             a1,
             a2,
             a3,
             a4,
             a5,
             (unsigned int)a6);
  if ( !*(_QWORD *)(a8 + 32) )
    sub_426248((__int64)"uninitialized __any_string");
  v12 = *(_BYTE **)a8;
  v13 = *(_QWORD *)(a8 + 8);
  v17[0] = (__int64)v18;
  sub_221FC40(v17, v12, (__int64)&v12[v13]);
  v14 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 24LL))(
          a1,
          a2,
          a3,
          a4,
          a5,
          (unsigned int)a6,
          v17);
  if ( (_BYTE *)v17[0] != v18 )
    j___libc_free_0(v17[0]);
  return v14;
}
