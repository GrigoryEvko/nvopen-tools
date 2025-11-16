// Function: sub_D305E0
// Address: 0xd305e0
//
char __fastcall sub_D305E0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  char result; // al
  unsigned __int64 v15; // [rsp+0h] [rbp-60h]
  char v17; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  _QWORD v20[8]; // [rsp+20h] [rbp-40h] BYREF

  v11 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v11 != 12
    && (unsigned __int8)v11 > 3u
    && (_BYTE)v11 != 5
    && (v11 & 0xFB) != 0xA
    && (v11 & 0xFD) != 4
    && ((unsigned __int8)(v11 - 15) > 3u && v11 != 20 || !(unsigned __int8)sub_BCEBA0(a2, 0))
    || sub_BCEA30(a2) )
  {
    return 0;
  }
  v12 = sub_9208B0(a4, a2);
  v20[1] = v13;
  v20[0] = (unsigned __int64)(v12 + 7) >> 3;
  v15 = sub_CA1930(v20);
  v19 = sub_AE43A0(a4, *(_QWORD *)(a1 + 8));
  if ( v19 > 0x40 )
    sub_C43690((__int64)&v18, v15, 0);
  else
    v18 = v15;
  result = sub_D30550((unsigned __int8 *)a1, a3, &v18, a4, a5, a6, a7, a8);
  if ( v19 > 0x40 )
  {
    if ( v18 )
    {
      v17 = result;
      j_j___libc_free_0_0(v18);
      return v17;
    }
  }
  return result;
}
