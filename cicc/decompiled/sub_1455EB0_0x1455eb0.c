// Function: sub_1455EB0
// Address: 0x1455eb0
//
__int64 __fastcall sub_1455EB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char v4; // r8
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdi

  v3 = 0x17FFFFFFE8LL;
  v4 = *(_BYTE *)(a1 + 23) & 0x40;
  v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v5 )
  {
    v6 = 24LL * *(unsigned int *)(a1 + 56) + 8;
    v7 = 0;
    do
    {
      v8 = a1 - 24LL * v5;
      if ( v4 )
        v8 = *(_QWORD *)(a1 - 8);
      if ( a2 == *(_QWORD *)(v8 + v6) )
      {
        v3 = 24 * v7;
        goto LABEL_8;
      }
      ++v7;
      v6 += 8;
    }
    while ( v5 != (_DWORD)v7 );
    v3 = 0x17FFFFFFE8LL;
  }
LABEL_8:
  if ( v4 )
    v9 = *(_QWORD *)(a1 - 8);
  else
    v9 = a1 - 24LL * v5;
  return *(_QWORD *)(v9 + v3);
}
