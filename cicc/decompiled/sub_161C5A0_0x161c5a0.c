// Function: sub_161C5A0
// Address: 0x161c5a0
//
__int64 __fastcall sub_161C5A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  _QWORD v20[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+28h] [rbp-38h]

  v7 = sub_1643360(*a1);
  v20[0] = a2;
  v20[1] = a3;
  v10 = sub_159C470(v7, a4, 0);
  if ( a5 )
  {
    v21 = sub_161BD20((__int64)a1, v10, v8, v9);
    v11 = sub_159C470(v7, 1, 0);
    v14 = sub_161BD20((__int64)a1, v11, v12, v13);
    v15 = *a1;
    v22 = v14;
    return sub_1627350(v15, v20, 4, 0, 1);
  }
  else
  {
    v17 = sub_161BD20((__int64)a1, v10, v8, v9);
    v18 = *a1;
    v21 = v17;
    return sub_1627350(v18, v20, 3, 0, 1);
  }
}
