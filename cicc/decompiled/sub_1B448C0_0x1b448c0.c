// Function: sub_1B448C0
// Address: 0x1b448c0
//
bool __fastcall sub_1B448C0(__int64 **a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // r8
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  bool result; // al
  __int64 v9; // rax
  char v10; // r10
  unsigned int v11; // r9d
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx

  v4 = sub_1648700(*(_QWORD *)(a2 + 8));
  v5 = *(_QWORD *)(a2 + 40);
  v6 = v4;
  v7 = **a1;
  if ( !v7 || *a1[1] != *(_QWORD *)(v7 + 40) )
    return v6[5] == v5;
  v9 = 0x17FFFFFFE8LL;
  v10 = *(_BYTE *)(v7 + 23) & 0x40;
  v11 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
  if ( v11 )
  {
    v12 = 24LL * *(unsigned int *)(v7 + 56) + 8;
    v13 = 0;
    do
    {
      v14 = v7 - 24LL * v11;
      if ( v10 )
        v14 = *(_QWORD *)(v7 - 8);
      if ( *(_QWORD *)(v14 + v12) == v5 )
      {
        v9 = 24 * v13;
        goto LABEL_11;
      }
      ++v13;
      v12 += 8;
    }
    while ( v11 != (_DWORD)v13 );
    v9 = 0x17FFFFFFE8LL;
  }
LABEL_11:
  v15 = v10 ? *(_QWORD *)(v7 - 8) : v7 - 24LL * v11;
  result = a2 == *(_QWORD *)(v15 + v9) && *(_QWORD *)(v15 + v9) != 0;
  if ( !result )
    return v6[5] == v5;
  return result;
}
