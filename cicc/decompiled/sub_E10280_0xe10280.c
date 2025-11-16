// Function: sub_E10280
// Address: 0xe10280
//
__int64 __fastcall sub_E10280(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax

  v3 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(void **)a2;
  if ( v3 + 1 > v5 )
  {
    v7 = v3 + 993;
    v8 = 2 * v5;
    if ( v7 > v8 )
      *(_QWORD *)(a2 + 16) = v7;
    else
      *(_QWORD *)(a2 + 16) = v8;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (void *)v9;
    if ( !v9 )
      abort();
    v3 = *(_QWORD *)(a2 + 8);
  }
  *((_BYTE *)v6 + v3) = 126;
  ++*(_QWORD *)(a2 + 8);
  return (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 32LL))(*(_QWORD *)(a1 + 16), a2);
}
