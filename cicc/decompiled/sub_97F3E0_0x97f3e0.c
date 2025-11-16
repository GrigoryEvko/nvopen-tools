// Function: sub_97F3E0
// Address: 0x97f3e0
//
__int64 __fastcall sub_97F3E0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  _QWORD v4[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-40h] BYREF
  __int128 v6; // [rsp+20h] [rbp-30h]
  __int64 v7; // [rsp+30h] [rbp-20h]

  v2 = a1 + 8;
  *(_QWORD *)(v2 - 8) = 0;
  *(_QWORD *)(v2 + 115) = 0;
  *(_QWORD *)(v2 + 128) = 0;
  *(_QWORD *)(v2 + 136) = 0;
  *(_QWORD *)(v2 + 144) = 0;
  *(_DWORD *)(v2 + 152) = 0;
  *(_QWORD *)(v2 + 168) = 0;
  *(_QWORD *)(v2 + 176) = 0;
  *(_QWORD *)(v2 + 184) = 0;
  *(_QWORD *)(v2 + 192) = 0;
  *(_QWORD *)(v2 + 200) = 0;
  *(_QWORD *)(v2 + 208) = 0;
  memset(
    (void *)(v2 & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8 * ((unsigned __int64)((unsigned int)a1 - (v2 & 0xFFFFFFF8) + 131) >> 3));
  v4[0] = v5;
  v7 = 0;
  v4[1] = 0;
  LOBYTE(v5[0]) = 0;
  v6 = 0;
  result = sub_97DEE0(a1, (__int64)v4);
  if ( (_QWORD *)v4[0] != v5 )
    return j_j___libc_free_0(v4[0], v5[0] + 1LL);
  return result;
}
