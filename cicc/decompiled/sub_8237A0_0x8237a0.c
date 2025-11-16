// Function: sub_8237A0
// Address: 0x8237a0
//
__int64 *__fastcall sub_8237A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax

  v6 = (__int64 *)sub_822B10(40, a2, a3, a4, a5, a6);
  v6[1] = a1;
  v7 = v6;
  v6[3] = a1;
  v6[2] = 0;
  v12 = malloc(a1, a2, v8, v9, v10, v11);
  if ( !v12 )
    sub_685240(4u);
  v7[4] = v12;
  v13 = qword_4F195E8;
  qword_4F195E8 = (__int64)v7;
  *v7 = v13;
  return v7;
}
