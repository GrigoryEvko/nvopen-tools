// Function: sub_15A37D0
// Address: 0x15a37d0
//
__int64 __fastcall sub_15A37D0(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r14
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD v11[2]; // [rsp+0h] [rbp-60h] BYREF
  __int128 v12; // [rsp+10h] [rbp-50h]
  __int128 v13; // [rsp+20h] [rbp-40h]
  __int128 v14; // [rsp+30h] [rbp-30h]

  result = sub_1584570(a1, a2);
  if ( !result )
  {
    v5 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
    if ( a3 != v5 )
    {
      v11[0] = a1;
      v11[1] = a2;
      *(_QWORD *)&v12 = 59;
      *((_QWORD *)&v12 + 1) = v11;
      v13 = 2u;
      v14 = 0u;
      v6 = (_QWORD *)sub_16498A0(a1);
      return sub_15A2780(*v6 + 1776LL, v5, v7, v8, v9, v10, v12, v13, v14);
    }
  }
  return result;
}
