// Function: sub_3774B00
// Address: 0x3774b00
//
__int64 __fastcall sub_3774B00(__int64 *a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rcx
  __int64 v4; // r8
  _QWORD *v5; // r12
  int v6; // edx
  int v7; // r14d
  __int64 result; // rax
  __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  int v10; // [rsp+18h] [rbp-28h]

  v2 = (_QWORD *)a1[1];
  v3 = a1[2];
  v9 = 0;
  v4 = a1[3];
  v10 = 0;
  v5 = sub_33F17F0(v2, 51, (__int64)&v9, v3, v4);
  v7 = v6;
  if ( v9 )
    sub_B91220((__int64)&v9, v9);
  result = *a1;
  *(_QWORD *)result = v5;
  *(_DWORD *)(result + 8) = v7;
  return result;
}
