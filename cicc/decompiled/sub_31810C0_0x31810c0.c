// Function: sub_31810C0
// Address: 0x31810c0
//
__int64 __fastcall sub_31810C0(unsigned __int64 *a1, __int64 a2, char a3)
{
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // r15
  __int128 v7; // rax
  __int64 v8; // rax
  size_t v9; // rdx
  char v10; // cl
  int *v11; // rsi
  __int64 v13; // rax
  __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v14[0] = sub_B2D7E0(a2, "sample-profile-suffix-elision-policy", 0x24u);
  v4 = sub_A72240(v14);
  v6 = v5;
  *(_QWORD *)&v7 = sub_BD5D20(a2);
  v8 = sub_C16140(v7, v4, v6);
  v10 = a3;
  v11 = (int *)v8;
  if ( v9 && unk_4F838D1 )
  {
    v13 = sub_B2F650(v8, v9);
    v10 = a3;
    v11 = 0;
    v9 = v13;
  }
  return sub_3180E70(a1, v11, v9, v10);
}
