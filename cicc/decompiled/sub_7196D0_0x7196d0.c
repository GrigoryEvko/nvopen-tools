// Function: sub_7196D0
// Address: 0x7196d0
//
__int64 __fastcall sub_7196D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v11[4]; // [rsp+10h] [rbp-20h] BYREF

  v10 = 0;
  if ( !word_4D04898 )
    return 0;
  v10 = sub_724DC0(a1, a2, &word_4D04898, a4, a5, a6);
  v11[0] = 0;
  v11[1] = 0;
  if ( (unsigned int)sub_7A30C0(a1, 0, 1, v10) )
    v10 = sub_724E50(&v10, 0, v7, v8, v9);
  else
    sub_724E30(&v10);
  sub_67E3D0(v11);
  return v10;
}
