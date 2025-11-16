// Function: sub_E29DE0
// Address: 0xe29de0
//
__int64 __fastcall sub_E29DE0(__int64 a1, __int64 a2)
{
  _DWORD *v4; // rdi
  int v5; // eax
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  void *v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rax

  v4 = *(_DWORD **)(a1 + 32);
  v5 = v4[2];
  if ( v5 == 3 || v5 == 16 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(_QWORD *)(a2 + 16);
    v9 = *(void **)a2;
    if ( v7 + 1 > v8 )
    {
      v10 = v7 + 993;
      v11 = 2 * v8;
      if ( v10 > v11 )
        *(_QWORD *)(a2 + 16) = v10;
      else
        *(_QWORD *)(a2 + 16) = v11;
      v12 = realloc(v9);
      *(_QWORD *)a2 = v12;
      v9 = (void *)v12;
      if ( !v12 )
        abort();
      v7 = *(_QWORD *)(a2 + 8);
    }
    *((_BYTE *)v9 + v7) = 41;
    ++*(_QWORD *)(a2 + 8);
    v4 = *(_DWORD **)(a1 + 32);
  }
  return (*(__int64 (__fastcall **)(_DWORD *, __int64))(*(_QWORD *)v4 + 32LL))(v4, a2);
}
