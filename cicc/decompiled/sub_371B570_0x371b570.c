// Function: sub_371B570
// Address: 0x371b570
//
__int64 __fastcall sub_371B570(__int64 a1, __int64 a2)
{
  char v2; // r15
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v11; // rax

  v2 = 1;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_QWORD *)(v4 + 56);
  if ( (*(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL) != v4 + 48 )
  {
    v6 = v5 - 24;
    if ( !v5 )
      v6 = 0;
    v7 = sub_3186770(*(_QWORD *)(a2 + 24), v6);
    v8 = (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v7 + 88LL))(v7) - 1;
    if ( v8 )
    {
      v9 = v8 - 1;
      do
        v5 = *(_QWORD *)(v5 + 8);
      while ( v9-- != 0 );
      v2 = 0;
    }
  }
  v11 = *(_QWORD *)(a2 + 24);
  *(_QWORD *)a1 = v4;
  *(_QWORD *)(a1 + 8) = v5;
  *(_BYTE *)(a1 + 16) = v2;
  *(_QWORD *)(a1 + 24) = v11;
  *(_BYTE *)(a1 + 17) = 0;
  return a1;
}
