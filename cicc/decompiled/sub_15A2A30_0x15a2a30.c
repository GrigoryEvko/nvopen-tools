// Function: sub_15A2A30
// Address: 0x15a2a30
//
__int64 __fastcall sub_15A2A30(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  char v9; // r13
  __int64 result; // rax
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD v17[2]; // [rsp+0h] [rbp-70h] BYREF
  __int128 v18; // [rsp+10h] [rbp-60h]
  __int128 v19; // [rsp+20h] [rbp-50h]
  __int128 v20; // [rsp+30h] [rbp-40h]

  v9 = a4;
  result = sub_1584E20(a1, (__int64)a2, a3, a4, a6, a7, a8);
  if ( !result && a5 != *a2 )
  {
    v17[0] = a2;
    WORD1(v18) = 0;
    v17[1] = a3;
    LOBYTE(v18) = (_BYTE)a1;
    BYTE1(v18) = v9;
    *((_QWORD *)&v18 + 1) = v17;
    v19 = 2u;
    v20 = 0u;
    v12 = (_QWORD *)sub_16498A0(a2);
    return sub_15A2780(*v12 + 1776LL, *a2, v13, v14, v15, v16, v18, v19, v20);
  }
  return result;
}
