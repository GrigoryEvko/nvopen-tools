// Function: sub_16A5C50
// Address: 0x16a5c50
//
__int64 __fastcall sub_16A5C50(__int64 a1, const void **a2, unsigned int a3)
{
  unsigned __int64 v4; // rsi
  unsigned __int64 v6; // r12
  void *v7; // rcx
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  char *v10; // [rsp+8h] [rbp-38h]

  if ( a3 > 0x40 )
  {
    v6 = ((unsigned __int64)a3 + 63) >> 6;
    v7 = (void *)sub_2207820(8 * v6);
    v8 = *((unsigned int *)a2 + 2);
    v9 = (unsigned __int64)(v8 + 63) >> 6;
    if ( (unsigned int)v8 > 0x40 )
      a2 = (const void **)*a2;
    v10 = (char *)memcpy(v7, a2, 8 * v9);
    memset(&v10[8 * v9], 0, (unsigned int)(8 * (v6 - v9)));
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v10;
  }
  else
  {
    v4 = (unsigned __int64)*a2;
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v4 & (0xFFFFFFFFFFFFFFFFLL >> -(char)a3);
  }
  return a1;
}
