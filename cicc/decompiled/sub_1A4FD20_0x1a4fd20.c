// Function: sub_1A4FD20
// Address: 0x1a4fd20
//
bool __fastcall sub_1A4FD20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  __int64 v6; // r9
  __int64 v7; // rax
  char v8; // di
  unsigned int v9; // esi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r9
  bool result; // al

  for ( i = *(_QWORD *)(a3 + 48); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v6 = i - 24;
    v7 = 0x17FFFFFFE8LL;
    v8 = *(_BYTE *)(i - 1) & 0x40;
    v9 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v9 )
    {
      v10 = 24LL * *(unsigned int *)(i + 32) + 8;
      v11 = 0;
      do
      {
        v12 = v6 - 24LL * v9;
        if ( v8 )
          v12 = *(_QWORD *)(i - 32);
        if ( a2 == *(_QWORD *)(v12 + v10) )
        {
          v7 = 24 * v11;
          goto LABEL_11;
        }
        ++v11;
        v10 += 8;
      }
      while ( v9 != (_DWORD)v11 );
      v7 = 0x17FFFFFFE8LL;
    }
LABEL_11:
    if ( v8 )
      v13 = *(_QWORD *)(i - 32);
    else
      v13 = v6 - 24LL * v9;
    result = sub_13FC1A0(a1, *(_QWORD *)(v13 + v7));
    if ( !result )
      return result;
  }
  return 1;
}
