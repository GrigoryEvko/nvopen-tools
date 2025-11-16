// Function: sub_AA5D60
// Address: 0xaa5d60
//
void __fastcall sub_AA5D60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rcx
  __int64 i; // r8
  __int64 v8; // rdx
  __int64 v9; // rsi
  _QWORD *v10; // rax

  v6 = *(_QWORD *)(a1 + 56);
  for ( i = a1 + 48; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)(v6 - 24) != 84 )
      break;
    if ( (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) != 0 )
    {
      v8 = 0;
      v9 = 8LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF);
      do
      {
        while ( 1 )
        {
          v10 = (_QWORD *)(*(_QWORD *)(v6 - 32) + v8 + 32LL * *(unsigned int *)(v6 + 48));
          if ( a2 == *v10 )
            break;
          v8 += 8;
          if ( v8 == v9 )
            goto LABEL_9;
        }
        v8 += 8;
        *v10 = a3;
      }
      while ( v8 != v9 );
    }
LABEL_9:
    ;
  }
}
