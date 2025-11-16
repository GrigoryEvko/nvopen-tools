// Function: sub_C449B0
// Address: 0xc449b0
//
__int64 __fastcall sub_C449B0(__int64 a1, const void **a2, unsigned int a3)
{
  char *v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // r13
  const void *v8; // rax
  unsigned __int64 v9; // [rsp+8h] [rbp-38h]

  if ( a3 <= 0x40 )
  {
    v8 = *a2;
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v8;
  }
  else if ( *((_DWORD *)a2 + 2) == a3 )
  {
    *(_DWORD *)(a1 + 8) = a3;
    sub_C43780(a1, a2);
  }
  else
  {
    v9 = ((unsigned __int64)a3 + 63) >> 6;
    v4 = (char *)sub_2207820(8 * v9);
    v5 = *((unsigned int *)a2 + 2);
    v6 = (unsigned __int64)(v5 + 63) >> 6;
    if ( (unsigned int)v5 > 0x40 )
      a2 = (const void **)*a2;
    memcpy(v4, a2, 8 * v6);
    memset(&v4[8 * v6], 0, (unsigned int)(8 * (v9 - v6)));
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v4;
  }
  return a1;
}
