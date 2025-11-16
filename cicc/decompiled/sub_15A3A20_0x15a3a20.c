// Function: sub_15A3A20
// Address: 0x15a3a20
//
__int64 __fastcall sub_15A3A20(__int64 *a1, __int64 *a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  __int64 result; // rax
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD v15[2]; // [rsp+10h] [rbp-70h] BYREF
  __int128 v16; // [rsp+20h] [rbp-60h]
  __int128 v17; // [rsp+30h] [rbp-50h]
  __int128 v18; // [rsp+40h] [rbp-40h]

  v7 = *a2;
  result = sub_1584C50(a1, (__int64)a2, a3, a4);
  if ( !result && v7 != a5 )
  {
    v15[0] = a1;
    v15[1] = a2;
    *((_QWORD *)&v17 + 1) = a3;
    v18 = (unsigned __int64)a4;
    *(_QWORD *)&v16 = 63;
    *((_QWORD *)&v16 + 1) = v15;
    *(_QWORD *)&v17 = 2;
    v9 = (_QWORD *)sub_16498A0(a1);
    return sub_15A2780(*v9 + 1776LL, v7, v10, v11, v12, v13, v16, v17, v18);
  }
  return result;
}
