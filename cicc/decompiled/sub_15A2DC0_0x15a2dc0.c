// Function: sub_15A2DC0
// Address: 0x15a2dc0
//
__int64 __fastcall sub_15A2DC0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD v12[4]; // [rsp+0h] [rbp-70h] BYREF
  __int128 v13; // [rsp+20h] [rbp-50h]
  __int128 v14; // [rsp+30h] [rbp-40h]
  __int128 v15; // [rsp+40h] [rbp-30h]

  result = sub_1584040(a1, a2, a3);
  if ( !result && a4 != *a2 )
  {
    v12[0] = a1;
    v12[1] = a2;
    v12[2] = a3;
    *(_QWORD *)&v13 = 55;
    *((_QWORD *)&v13 + 1) = v12;
    v14 = 3u;
    v15 = 0u;
    v7 = (_QWORD *)sub_16498A0(a1);
    return sub_15A2780(*v7 + 1776LL, *a2, v8, v9, v10, v11, v13, v14, v15);
  }
  return result;
}
