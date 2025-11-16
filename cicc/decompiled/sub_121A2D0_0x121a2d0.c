// Function: sub_121A2D0
// Address: 0x121a2d0
//
__int64 __fastcall sub_121A2D0(_QWORD **a1, _QWORD *a2, char a3)
{
  _QWORD **v4; // rsi
  unsigned int v6; // r12d
  _QWORD *v8; // [rsp+0h] [rbp-80h] BYREF
  __int64 v9; // [rsp+8h] [rbp-78h]
  _BYTE v10[112]; // [rsp+10h] [rbp-70h] BYREF

  v4 = &v8;
  v8 = v10;
  v9 = 0x800000000LL;
  v6 = sub_121A060((__int64)a1, (__int64)&v8);
  if ( !(_BYTE)v6 )
  {
    v4 = (_QWORD **)v8;
    *a2 = sub_BD0B90(*a1, v8, (unsigned int)v9, a3);
  }
  if ( v8 != (_QWORD *)v10 )
    _libc_free(v8, v4);
  return v6;
}
