// Function: sub_37333F0
// Address: 0x37333f0
//
void __fastcall sub_37333F0(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi
  int *v6; // rax
  size_t v7; // rdx
  int v8[8]; // [rsp+Fh] [rbp-21h] BYREF

  v2 = *a2;
  v3 = a2[1];
  if ( v3 != *a2 )
  {
    do
    {
      if ( *(_DWORD *)(v2 + 8) == 5 )
      {
        v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 160) + 760LL) + 16LL * *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL) + 8);
        v6 = (int *)sub_372FC20(v5);
        sub_37333A0((int *)a1, v5, v6, v7);
      }
      else
      {
        LOBYTE(v8[0]) = *(_QWORD *)(v2 + 16);
        sub_C7D060((int *)a1, v8, 1u);
      }
      v4 = *(_QWORD *)v2;
      v2 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v4 & 4) != 0 )
        v2 = 0;
    }
    while ( v3 != v2 );
  }
}
