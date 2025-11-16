// Function: sub_E30FA0
// Address: 0xe30fa0
//
void __fastcall sub_E30FA0(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  void *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax

  if ( *(_BYTE *)(a1 + 32) )
  {
    v6 = a2[1];
    v7 = a2[2];
    v8 = (void *)*a2;
    if ( v6 + 1 > v7 )
    {
      v9 = v6 + 993;
      v10 = 2 * v7;
      if ( v9 > v10 )
        a2[2] = v9;
      else
        a2[2] = v10;
      v11 = realloc(v8);
      *a2 = v11;
      v8 = (void *)v11;
      if ( !v11 )
        abort();
      v6 = a2[1];
    }
    *((_BYTE *)v8 + v6) = 126;
    ++a2[1];
  }
  (*(void (__fastcall **)(_QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 24) + 16LL))(*(_QWORD *)(a1 + 24), a2, a3);
  sub_E2EB40(a1, (__int64)a2, a3);
}
