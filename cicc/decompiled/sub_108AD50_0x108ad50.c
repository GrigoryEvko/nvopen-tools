// Function: sub_108AD50
// Address: 0x108ad50
//
__int64 *__fastcall sub_108AD50(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *a2;
  *a2 = 0;
  v10[0] = v6;
  v7 = sub_22077B0(136);
  v8 = v7;
  if ( v7 )
    sub_1086000(v7, v10, a3, a4);
  if ( v10[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10[0] + 8LL))(v10[0]);
  *a1 = v8;
  return a1;
}
