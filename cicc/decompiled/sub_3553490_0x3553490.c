// Function: sub_3553490
// Address: 0x3553490
//
__int64 __fastcall sub_3553490(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rax

  v3 = a2 - a1;
  v5 = 0x2E8BA2E8BA2E8BA3LL * (v3 >> 3);
  if ( v3 <= 0 )
    return a3;
  v6 = a1 + 32;
  v7 = a3 + 32;
  do
  {
    sub_C7D6A0(*(_QWORD *)(v7 - 24), 8LL * *(unsigned int *)(v7 - 8), 8);
    *(_DWORD *)(v7 - 8) = 0;
    *(_QWORD *)(v7 - 24) = 0;
    *(_DWORD *)(v7 - 16) = 0;
    *(_DWORD *)(v7 - 12) = 0;
    ++*(_QWORD *)(v7 - 32);
    v9 = *(_QWORD *)(v6 - 24);
    ++*(_QWORD *)(v6 - 32);
    v10 = *(_QWORD *)(v7 - 24);
    *(_QWORD *)(v7 - 24) = v9;
    LODWORD(v9) = *(_DWORD *)(v6 - 16);
    *(_QWORD *)(v6 - 24) = v10;
    LODWORD(v10) = *(_DWORD *)(v7 - 16);
    *(_DWORD *)(v7 - 16) = v9;
    LODWORD(v9) = *(_DWORD *)(v6 - 12);
    *(_DWORD *)(v6 - 16) = v10;
    LODWORD(v10) = *(_DWORD *)(v7 - 12);
    *(_DWORD *)(v7 - 12) = v9;
    LODWORD(v9) = *(_DWORD *)(v6 - 8);
    *(_DWORD *)(v6 - 12) = v10;
    LODWORD(v10) = *(_DWORD *)(v7 - 8);
    *(_DWORD *)(v7 - 8) = v9;
    *(_DWORD *)(v6 - 8) = v10;
    if ( v7 != v6 )
    {
      if ( *(_DWORD *)(v6 + 8) )
      {
        if ( *(_QWORD *)v7 != v7 + 16 )
          _libc_free(*(_QWORD *)v7);
        *(_QWORD *)v7 = *(_QWORD *)v6;
        *(_DWORD *)(v7 + 8) = *(_DWORD *)(v6 + 8);
        *(_DWORD *)(v7 + 12) = *(_DWORD *)(v6 + 12);
        *(_QWORD *)v6 = v6 + 16;
        *(_DWORD *)(v6 + 12) = 0;
        *(_DWORD *)(v6 + 8) = 0;
      }
      else
      {
        *(_DWORD *)(v7 + 8) = 0;
      }
    }
    v8 = *(_BYTE *)(v6 + 16);
    v7 += 88;
    v6 += 88;
    *(_BYTE *)(v7 - 72) = v8;
    *(_DWORD *)(v7 - 68) = *(_DWORD *)(v6 - 68);
    *(_DWORD *)(v7 - 64) = *(_DWORD *)(v6 - 64);
    *(_DWORD *)(v7 - 60) = *(_DWORD *)(v6 - 60);
    *(_DWORD *)(v7 - 56) = *(_DWORD *)(v6 - 56);
    *(_QWORD *)(v7 - 48) = *(_QWORD *)(v6 - 48);
    *(_DWORD *)(v7 - 40) = *(_DWORD *)(v6 - 40);
    --v5;
  }
  while ( v5 );
  return a3 + v3;
}
