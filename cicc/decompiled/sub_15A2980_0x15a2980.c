// Function: sub_15A2980
// Address: 0x15a2980
//
__int64 __fastcall sub_15A2980(unsigned __int8 a1, unsigned __int64 a2, __int64 **a3, char a4)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int128 v13; // [rsp-30h] [rbp-90h]
  unsigned __int64 v14; // [rsp+8h] [rbp-58h] BYREF
  __int64 v15; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  __int64 v18; // [rsp+28h] [rbp-38h]
  __int64 v19; // [rsp+30h] [rbp-30h]
  __int64 v20; // [rsp+38h] [rbp-28h]

  v14 = a2;
  result = sub_1582CC0(a1, a2, a3);
  if ( !result && !a4 )
  {
    v11 = *a3;
    LODWORD(v15) = a1;
    v12 = *v11;
    v17 = 1;
    v16 = &v14;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    *((_QWORD *)&v13 + 1) = &v14;
    *(_QWORD *)&v13 = v15;
    return sub_15A2780(v12 + 1776, (__int64)a3, v7, v8, v9, v10, v13, 1u, 0);
  }
  return result;
}
