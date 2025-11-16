// Function: sub_2B63690
// Address: 0x2b63690
//
__int64 __fastcall sub_2B63690(_QWORD *a1, int a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r10
  _DWORD *v4; // rax
  unsigned int *v6[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE v7[16]; // [rsp+10h] [rbp-70h] BYREF
  __int64 **v8[2]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v9[80]; // [rsp+30h] [rbp-50h] BYREF

  v6[1] = (unsigned int *)0x400000000LL;
  v3 = a1[3];
  v8[1] = (__int64 **)0x600000000LL;
  v4 = (_DWORD *)a1[2];
  v6[0] = (unsigned int *)v7;
  v8[0] = (__int64 **)v9;
  LOBYTE(v2) = (unsigned int)sub_2B60A30(
                               v3,
                               (_QWORD *)(*a1 + 8LL * (unsigned int)(*v4 * a2)),
                               (unsigned int)*v4,
                               *(_QWORD *)(*a1 + 8LL * (unsigned int)(*v4 * a2)),
                               v6,
                               v8,
                               0,
                               1) == 2;
  if ( (_BYTE *)v8[0] != v9 )
    _libc_free((unsigned __int64)v8[0]);
  if ( (_BYTE *)v6[0] != v7 )
    _libc_free((unsigned __int64)v6[0]);
  return v2;
}
