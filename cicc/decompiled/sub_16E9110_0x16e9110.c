// Function: sub_16E9110
// Address: 0x16e9110
//
__int64 __fastcall sub_16E9110(_QWORD *a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // r14
  __int64 v7; // rdx
  signed __int64 v8; // rax
  __int64 v10; // r13
  signed __int64 v11; // rsi
  __int64 v12; // rdx

  v6 = a1[5];
  v7 = a3 - a2;
  if ( v7 )
  {
    v8 = a1[4];
    v10 = v7;
    v11 = v8 + v7;
    v12 = a1[5];
    if ( v8 < v11 )
    {
      sub_16E90A0((__int64)a1, v11, v12, a4, a5, a6);
      v12 = a1[5];
    }
    memmove((void *)(a1[3] + 8 * v12), (const void *)(a1[3] + 8 * a2), 8 * v10);
    a1[5] += v10;
  }
  return v6;
}
