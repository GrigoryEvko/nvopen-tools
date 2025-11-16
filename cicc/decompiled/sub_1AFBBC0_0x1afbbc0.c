// Function: sub_1AFBBC0
// Address: 0x1afbbc0
//
__int64 __fastcall sub_1AFBBC0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r11
  __int64 v5; // rdi
  __int64 v6; // rax
  char v7; // r8
  unsigned int v8; // esi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax

  v1 = sub_13FC520(a1);
  v2 = sub_157F280(**(_QWORD **)(a1 + 32));
  v4 = v3;
  v5 = v2;
  if ( v2 == v3 )
    return 0;
  while ( 1 )
  {
    v6 = 0x17FFFFFFE8LL;
    v7 = *(_BYTE *)(v5 + 23) & 0x40;
    v8 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
    if ( v8 )
    {
      v9 = 24LL * *(unsigned int *)(v5 + 56) + 8;
      v10 = 0;
      do
      {
        v11 = v5 - 24LL * v8;
        if ( v7 )
          v11 = *(_QWORD *)(v5 - 8);
        if ( v1 == *(_QWORD *)(v11 + v9) )
        {
          v6 = 24 * v10;
          goto LABEL_9;
        }
        ++v10;
        v9 += 8;
      }
      while ( v8 != (_DWORD)v10 );
      v6 = 0x17FFFFFFE8LL;
    }
LABEL_9:
    if ( !v7 )
      break;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v5 - 8) + v6) + 16LL) == 13 )
      return 1;
LABEL_11:
    v12 = *(_QWORD *)(v5 + 32);
    if ( !v12 )
      BUG();
    v5 = 0;
    if ( *(_BYTE *)(v12 - 8) == 77 )
      v5 = v12 - 24;
    if ( v4 == v5 )
      return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v5 - 24LL * v8 + v6) + 16LL) != 13 )
    goto LABEL_11;
  return 1;
}
