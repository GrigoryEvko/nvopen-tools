// Function: sub_15A3AE0
// Address: 0x15a3ae0
//
__int64 __fastcall sub_15A3AE0(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 result; // rax
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // [rsp+8h] [rbp-68h] BYREF
  __int128 v14; // [rsp+10h] [rbp-60h]
  __int128 v15; // [rsp+20h] [rbp-50h]
  __int128 v16; // [rsp+30h] [rbp-40h]

  v6 = sub_15FB2A0(*a1, a2, a3);
  result = sub_1584BD0((__int64)a1, a2, a3);
  if ( !result && v6 != a4 )
  {
    v13 = a1;
    *((_QWORD *)&v15 + 1) = a2;
    v16 = (unsigned __int64)a3;
    *(_QWORD *)&v14 = 62;
    *((_QWORD *)&v14 + 1) = &v13;
    *(_QWORD *)&v15 = 1;
    v8 = (_QWORD *)sub_16498A0(a1);
    return sub_15A2780(*v8 + 1776LL, v6, v9, v10, v11, v12, v14, v15, v16);
  }
  return result;
}
