// Function: sub_CB10A0
// Address: 0xcb10a0
//
__int64 __fastcall sub_CB10A0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rsi
  __int64 v9; // rax
  int v10; // ecx
  unsigned int v11; // eax
  unsigned int v12; // ebx
  __int64 v13; // r14
  __int64 v14; // rdi
  int v15; // ecx
  const char *v17; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+20h] [rbp-30h]
  char v19; // [rsp+21h] [rbp-2Fh]

  v8 = *(__int64 **)(a1 + 672);
  *(_DWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 608) = 0;
  if ( *(_DWORD *)(*v8 + 32) == 5 )
  {
    v9 = (v8[2] - v8[1]) >> 3;
    *(_DWORD *)(a1 + 664) = v9;
    LOBYTE(v10) = v9;
    v11 = (unsigned int)(v9 + 63) >> 6;
    v12 = v11;
    if ( v11 )
    {
      v13 = v11;
      v14 = 0;
      if ( *(_DWORD *)(a1 + 612) < v11 )
      {
        sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v11, 8u, a5, a6);
        v14 = 8LL * *(unsigned int *)(a1 + 608);
      }
      memset((void *)(*(_QWORD *)(a1 + 600) + v14), 0, 8 * v13);
      *(_DWORD *)(a1 + 608) += v12;
      v10 = *(_DWORD *)(a1 + 664);
    }
    v15 = v10 & 0x3F;
    if ( v15 )
      *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 608) - 8) &= ~(-1LL << v15);
  }
  else
  {
    v19 = 1;
    v17 = "expected sequence of bit values";
    v18 = 3;
    sub_CB1040(a1, v8, (__int64)&v17);
  }
  *a2 = 1;
  return 1;
}
