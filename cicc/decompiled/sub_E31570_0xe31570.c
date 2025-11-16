// Function: sub_E31570
// Address: 0xe31570
//
void __fastcall sub_E31570(__int64 a1, char a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax

  if ( !*(_BYTE *)(a1 + 49) && *(_BYTE *)(a1 + 48) )
  {
    v4 = *(_QWORD *)(a1 + 64);
    v5 = *(_QWORD *)(a1 + 72);
    v6 = *(void **)(a1 + 56);
    v7 = v4 + 1;
    if ( v4 + 1 > v5 )
    {
      v8 = v4 + 993;
      v9 = 2 * v5;
      if ( v8 > v9 )
        *(_QWORD *)(a1 + 72) = v8;
      else
        *(_QWORD *)(a1 + 72) = v9;
      v10 = realloc(v6);
      *(_QWORD *)(a1 + 56) = v10;
      v6 = (void *)v10;
      if ( !v10 )
        abort();
      v4 = *(_QWORD *)(a1 + 64);
      v7 = v4 + 1;
    }
    *(_QWORD *)(a1 + 64) = v7;
    *((_BYTE *)v6 + v4) = a2;
  }
}
