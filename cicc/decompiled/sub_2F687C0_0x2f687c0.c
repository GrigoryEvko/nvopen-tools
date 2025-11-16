// Function: sub_2F687C0
// Address: 0x2f687c0
//
__int64 __fastcall sub_2F687C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v7; // rax
  _QWORD *v8; // rbx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 *v11; // [rsp+8h] [rbp-38h]

  v7 = *a2;
  a2[10] += 128;
  v8 = (_QWORD *)((v7 + 15) & 0xFFFFFFFFFFFFFFF0LL);
  if ( a2[1] >= (unsigned __int64)(v8 + 16) && v7 )
  {
    *a2 = (__int64)(v8 + 16);
    if ( !v8 )
    {
      MEMORY[0x68] = *(_QWORD *)(a1 + 104);
      BUG();
    }
  }
  else
  {
    v11 = a5;
    v10 = sub_9D1E70((__int64)a2, 128, 128, 4);
    a5 = v11;
    v8 = (_QWORD *)v10;
  }
  *v8 = v8 + 2;
  v8[1] = 0x200000000LL;
  v8[8] = v8 + 10;
  v8[9] = 0x200000000LL;
  v8[12] = 0;
  sub_2F68500((__int64)v8, a5, a2, a4, (__int64)a5);
  v8[13] = 0;
  v8[14] = a3;
  v8[15] = a4;
  result = *(_QWORD *)(a1 + 104);
  v8[13] = result;
  *(_QWORD *)(a1 + 104) = v8;
  return result;
}
