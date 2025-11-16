// Function: sub_1928010
// Address: 0x1928010
//
__int64 __fastcall sub_1928010(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int *v3; // rbx
  __int64 v4; // rax
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7[3]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = (unsigned int *)(a1 - 1);
  v4 = *a1;
  v7[0] = a2;
  v7[1] = a3;
  v8[0] = v4;
  while ( (unsigned __int8)sub_1921830(v7, (int *)v8, v3) )
  {
    v5 = *v3;
    v3 -= 2;
    v3[4] = v5;
    v3[5] = v3[3];
  }
  result = v8[0];
  *((_QWORD *)v3 + 1) = v8[0];
  return result;
}
