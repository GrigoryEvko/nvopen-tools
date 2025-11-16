// Function: sub_3704F20
// Address: 0x3704f20
//
unsigned __int64 *__fastcall sub_3704F20(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  unsigned int v7; // ecx
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_3714340(v9, a2 + 16, a3, a4);
  v5 = v9[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v6 = *(_QWORD *)(a2 + 8),
        v7 = *(_DWORD *)(v6 + 56) - *(_DWORD *)(a2 + 104),
        *(_QWORD *)(v6 + 56) = *(unsigned int *)(a2 + 104),
        sub_1254950((unsigned __int64 *)v9, *(_QWORD *)(a2 + 8), a3 + 8, v7),
        v5 = v9[0] & 0xFFFFFFFFFFFFFFFELL,
        (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    v9[0] = 0;
    *a1 = v5 | 1;
    sub_9C66B0(v9);
    return a1;
  }
  else
  {
    v9[0] = 0;
    sub_9C66B0(v9);
    *a1 = 1;
    sub_9C66B0(v9);
    return a1;
  }
}
