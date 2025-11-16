// Function: sub_1248FB0
// Address: 0x1248fb0
//
__int64 __fastcall sub_1248FB0(__int64 a1)
{
  __int64 *v1; // rsi
  unsigned int v2; // r12d
  unsigned int v4; // [rsp+4h] [rbp-5Ch] BYREF
  __int64 *v5; // [rsp+8h] [rbp-58h] BYREF
  int *v6[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v7[64]; // [rsp+20h] [rbp-40h] BYREF

  v1 = (__int64 *)&v5;
  v6[0] = (int *)v7;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v4 = -1;
  v6[1] = (int *)0xC00000000LL;
  if ( (unsigned __int8)sub_1245840((_QWORD **)a1, (__int64 *)&v5, 1u, (int *)&v4, (__int64)v6)
    || (v1 = v5, (unsigned __int8)sub_122E910(a1, (__int64)v5)) )
  {
    v2 = 1;
  }
  else
  {
    v1 = v5;
    v2 = sub_1248DF0(a1, (__int64)v5, v4, v6[0]);
  }
  if ( (_BYTE *)v6[0] != v7 )
    _libc_free(v6[0], v1);
  return v2;
}
