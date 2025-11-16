// Function: sub_143B730
// Address: 0x143b730
//
void __fastcall sub_143B730(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  __int64 v5; // rsi
  _QWORD *v6; // rax
  int v7; // r8d
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 16) > 0x17u )
  {
    v12 = a1;
    v4 = *(_QWORD **)a2;
    v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    v6 = sub_143B670(v4, v5, &v12);
    if ( (_QWORD *)v5 == v6 )
    {
      if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
      {
        v8 = 0;
        v9 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        do
        {
          if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
            v10 = *(_QWORD *)(a1 - 8);
          else
            v10 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          v11 = *(_QWORD *)(v10 + v8);
          if ( *(_BYTE *)(v11 + 16) > 0x17u )
            sub_143B730(v11, a2);
          v8 += 24;
        }
        while ( v9 != v8 );
      }
    }
    else
    {
      if ( (_QWORD *)v5 != v6 + 1 )
      {
        memmove(v6, v6 + 1, v5 - (_QWORD)(v6 + 1));
        v7 = *(_DWORD *)(a2 + 8);
      }
      *(_DWORD *)(a2 + 8) = v7 - 1;
    }
  }
}
