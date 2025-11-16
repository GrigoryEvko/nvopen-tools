// Function: sub_DC5140
// Address: 0xdc5140
//
_QWORD *__fastcall sub_DC5140(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  __int64 v9; // [rsp+8h] [rbp-38h]

  v9 = sub_D95540(a2);
  v6 = sub_D97050(a1, v9);
  if ( v6 == sub_D97050(a1, a3) )
    return (_QWORD *)a2;
  v7 = sub_D97050(a1, v9);
  if ( v7 <= sub_D97050(a1, a3) )
    return sub_DC5000(a1, a2, a3, a4);
  else
    return (_QWORD *)sub_DC5200(a1, a2, a3, a4);
}
