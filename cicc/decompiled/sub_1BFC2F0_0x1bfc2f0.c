// Function: sub_1BFC2F0
// Address: 0x1bfc2f0
//
__int64 __fastcall sub_1BFC2F0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  unsigned int v7; // r13d
  __int64 result; // rax
  unsigned int v9; // r13d
  __int64 v10; // r14
  unsigned int v11; // eax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 16);
  if ( *(_DWORD *)(a1 + 16) < v7 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    if ( v7 <= (unsigned __int64)(v10 << 6) )
      goto LABEL_6;
    v12 = *(_QWORD *)a1;
    v13 = (v7 + 63) >> 6;
    if ( v13 < 2 * v10 )
      v13 = 2 * v10;
    v14 = (__int64)realloc(v12, 8 * v13, 8 * (int)v13, a4, a5, a6);
    if ( !v14 && (8 * v13 || (v14 = malloc(1u)) == 0) )
    {
      v17 = v14;
      sub_16BD1C0("Allocation failed", 1u);
      v14 = v17;
    }
    *(_QWORD *)a1 = v14;
    *(_QWORD *)(a1 + 8) = v13;
    sub_13A4C60(a1, 0);
    v15 = *(_QWORD *)(a1 + 8) - (unsigned int)v10;
    if ( v15 )
      memset((void *)(*(_QWORD *)a1 + 8LL * (unsigned int)v10), 0, 8 * v15);
    v16 = *(_DWORD *)(a1 + 16);
    if ( v7 > v16 )
    {
LABEL_6:
      sub_13A4C60(a1, 0);
      v11 = *(_DWORD *)(a1 + 16);
      *(_DWORD *)(a1 + 16) = v7;
      if ( v7 < v11 )
LABEL_15:
        sub_13A4C60(a1, 0);
    }
    else
    {
      *(_DWORD *)(a1 + 16) = v7;
      if ( v7 < v16 )
        goto LABEL_15;
    }
    v7 = *(_DWORD *)(a2 + 16);
  }
  result = 0;
  v9 = (v7 + 63) >> 6;
  if ( v9 )
  {
    do
    {
      *(_QWORD *)(*(_QWORD *)a1 + 8 * result) |= *(_QWORD *)(*(_QWORD *)a2 + 8 * result);
      ++result;
    }
    while ( v9 != result );
  }
  return result;
}
