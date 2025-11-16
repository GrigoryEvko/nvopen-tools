// Function: sub_1593ED0
// Address: 0x1593ed0
//
__int64 __fastcall sub_1593ED0(__int64 a1)
{
  unsigned int v1; // r15d
  unsigned __int8 v2; // al
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r13
  char v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-38h]

  v1 = 1;
  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 > 3u )
  {
    v4 = a1;
    while ( v2 == 4 )
    {
      v4 = *(_QWORD *)(v4 - 48);
      v2 = *(_BYTE *)(v4 + 16);
      if ( v2 <= 3u )
        return 1;
    }
    v5 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    if ( v2 != 5 || *(_WORD *)(v4 + 18) != 13 )
      goto LABEL_9;
    v11 = *(_QWORD *)(v4 - 24LL * (unsigned int)v5);
    if ( *(_BYTE *)(v11 + 16) != 5 )
      v11 = 0;
    v12 = *(_QWORD *)(v4 + 24 * (1LL - (unsigned int)v5));
    if ( !v12 )
      BUG();
    if ( *(_BYTE *)(v12 + 16) == 5
      && v11
      && *(_WORD *)(v11 + 18) == 45
      && *(_WORD *)(v12 + 18) == 45
      && (v13 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)), *(_BYTE *)(v13 + 16) == 4)
      && (v14 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)), *(_BYTE *)(v14 + 16) == 4)
      && *(_QWORD *)(v14 - 48) == *(_QWORD *)(v13 - 48) )
    {
      return 0;
    }
    else
    {
LABEL_9:
      v1 = 0;
      if ( (_DWORD)v5 )
      {
        v6 = 0;
        v7 = 24 * v5;
        v8 = *(_BYTE *)(v4 + 23) & 0x40;
        v15 = v4 - 24 * v5;
        do
        {
          v9 = v15;
          if ( v8 )
            v9 = *(_QWORD *)(v4 - 8);
          v10 = *(_QWORD *)(v9 + v6);
          v6 += 24;
          v1 |= sub_1593ED0(v10);
        }
        while ( v6 != v7 );
      }
    }
  }
  return v1;
}
