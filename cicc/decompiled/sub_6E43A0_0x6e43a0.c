// Function: sub_6E43A0
// Address: 0x6e43a0
//
__int64 __fastcall sub_6E43A0(__int64 *a1, _QWORD *a2, _QWORD *a3, __int64 *a4, _QWORD *a5)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v11 = *a1;
  v8 = sub_6E3DA0(*a1, 0);
  v9 = v8;
  *a2 = *(_QWORD *)(v11 + 56);
  if ( a4 )
    *a4 = sub_6E3F00(*(_QWORD *)(v8 + 376), (__int64)a1, v8);
  result = *(_QWORD *)(v9 + 356);
  *a3 = result;
  if ( a5 )
  {
    result = *(_QWORD *)(v9 + 368);
    *a5 = result;
  }
  return result;
}
