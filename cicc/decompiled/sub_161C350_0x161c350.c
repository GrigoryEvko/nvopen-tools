// Function: sub_161C350
// Address: 0x161c350
//
__int64 __fastcall sub_161C350(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  _QWORD v10[5]; // [rsp-28h] [rbp-28h] BYREF

  if ( a3 == a2 )
    return 0;
  v10[0] = sub_161BD20((__int64)a1, a2, a3, a4);
  v7 = sub_161BD20((__int64)a1, a3, v5, v6);
  v8 = *a1;
  v10[1] = v7;
  return sub_1627350(v8, v10, 2, 0, 1);
}
