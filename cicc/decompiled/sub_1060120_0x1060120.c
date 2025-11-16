// Function: sub_1060120
// Address: 0x1060120
//
__int128 ***__fastcall sub_1060120(
        __int128 ***a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        char *a9,
        const char *a10)
{
  __int64 v13; // rax
  __int128 **v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rcx

  v13 = sub_22077B0(880);
  v14 = (__int128 **)v13;
  if ( !v13 )
  {
    if ( (unsigned __int8)sub_10600F0(0, 0, a2, a4, a5, a6, a7, a8, a9, a10) )
    {
      *a1 = 0;
      return a1;
    }
    goto LABEL_6;
  }
  sub_BA8740(v13, a9, (__int64)a10, a3);
  if ( !(unsigned __int8)sub_10600F0(v14, 0, a2, a4, a5, a6, a7, a8, a9, a10) )
  {
LABEL_6:
    *a1 = v14;
    return a1;
  }
  *a1 = 0;
  sub_BA9C10((_QWORD **)v14, 0, v15, v16);
  j_j___libc_free_0(v14, 880);
  return a1;
}
