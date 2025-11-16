// Function: sub_373B220
// Address: 0x373b220
//
__int64 __fastcall sub_373B220(__int64 *a1, unsigned __int64 **a2, __int64 a3)
{
  __int64 v4; // rdx
  _QWORD *v5; // rax
  __int64 v7[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = a1[11];
  a1[21] += 16;
  v5 = (_QWORD *)((v4 + 15) & 0xFFFFFFFFFFFFFFF0LL);
  if ( a1[12] < (unsigned __int64)(v5 + 2) || !v4 )
  {
    v5 = (_QWORD *)sub_9D1E70((__int64)(a1 + 11), 16, 16, 4);
    goto LABEL_4;
  }
  a1[11] = (__int64)(v5 + 2);
  if ( v5 )
  {
LABEL_4:
    *v5 = a1;
    v5[1] = a3;
  }
  v7[1] = (__int64)v5;
  v7[0] = 0xF000000000005LL;
  return sub_3248F80(a2, a1 + 11, v7);
}
