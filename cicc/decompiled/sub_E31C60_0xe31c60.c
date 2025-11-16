// Function: sub_E31C60
// Address: 0xe31c60
//
void __fastcall sub_E31C60(__int64 a1, size_t a2, const void *a3)
{
  __int64 v6; // rdx
  size_t v7; // rax
  char *v8; // rdi
  size_t v9; // rsi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rax

  if ( !*(_BYTE *)(a1 + 49) && *(_BYTE *)(a1 + 48) && a2 )
  {
    v6 = *(_QWORD *)(a1 + 64);
    v7 = *(_QWORD *)(a1 + 72);
    v8 = *(char **)(a1 + 56);
    v9 = v6 + a2;
    if ( v9 > v7 )
    {
      v10 = v9 + 992;
      v11 = 2 * v7;
      if ( v10 > v11 )
        *(_QWORD *)(a1 + 72) = v10;
      else
        *(_QWORD *)(a1 + 72) = v11;
      v12 = realloc(v8);
      *(_QWORD *)(a1 + 56) = v12;
      v8 = (char *)v12;
      if ( !v12 )
        abort();
      v6 = *(_QWORD *)(a1 + 64);
    }
    memcpy(&v8[v6], a3, a2);
    *(_QWORD *)(a1 + 64) += a2;
  }
}
