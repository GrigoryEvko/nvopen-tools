// Function: sub_E2A040
// Address: 0xe2a040
//
void __fastcall sub_E2A040(__int64 a1)
{
  __int64 v1; // r12
  void *v2; // r13
  int v4; // edi
  int v5; // eax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 )
  {
    v2 = *(void **)a1;
    v4 = *(char *)(*(_QWORD *)a1 + v1 - 1);
    v5 = isalnum(v4);
    if ( (_BYTE)v4 == 62 || v5 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      if ( v1 + 1 > v6 )
      {
        v7 = 2 * v6;
        if ( v1 + 993 > v7 )
          *(_QWORD *)(a1 + 16) = v1 + 993;
        else
          *(_QWORD *)(a1 + 16) = v7;
        v8 = realloc(v2);
        *(_QWORD *)a1 = v8;
        v2 = (void *)v8;
        if ( !v8 )
          abort();
        v1 = *(_QWORD *)(a1 + 8);
      }
      *((_BYTE *)v2 + v1) = 32;
      ++*(_QWORD *)(a1 + 8);
    }
  }
}
