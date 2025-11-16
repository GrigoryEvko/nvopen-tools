// Function: sub_1974500
// Address: 0x1974500
//
__int64 __fastcall sub_1974500(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rax

  result = sub_157F280(a1);
  if ( result != v6 )
  {
    v7 = v6;
    v8 = result;
    while ( 1 )
    {
      v9 = 0;
      v10 = 8LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
      if ( (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) != 0 )
        break;
LABEL_10:
      result = *(_QWORD *)(v8 + 32);
      if ( !result )
        BUG();
      v8 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v8 = result - 24;
      if ( v7 == v8 )
        return result;
    }
    while ( 1 )
    {
      v12 = 24LL * *(unsigned int *)(v8 + 56);
      if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
      {
        v11 = (_QWORD *)(*(_QWORD *)(v8 - 8) + v9 + v12 + 8);
        if ( a2 != *v11 )
          goto LABEL_6;
LABEL_9:
        v9 += 8;
        *v11 = a3;
        if ( v9 == v10 )
          goto LABEL_10;
      }
      else
      {
        v11 = (_QWORD *)(v8 + v9 + v12 + 8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
        if ( a2 == *v11 )
          goto LABEL_9;
LABEL_6:
        v9 += 8;
        if ( v9 == v10 )
          goto LABEL_10;
      }
    }
  }
  return result;
}
