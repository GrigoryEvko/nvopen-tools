// Function: sub_D76CA0
// Address: 0xd76ca0
//
unsigned __int64 __fastcall sub_D76CA0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  _QWORD *i; // rcx
  __int64 v11; // rdx

  v4 = sub_B2F650(a2, a3);
  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    v6 = a1 + 8;
    do
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(result + 16);
        v8 = *(_QWORD *)(result + 24);
        if ( v4 <= *(_QWORD *)(result + 32) )
          break;
        result = *(_QWORD *)(result + 24);
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = result;
      result = *(_QWORD *)(result + 16);
    }
    while ( v7 );
LABEL_6:
    if ( a1 + 8 != v6 && v4 >= *(_QWORD *)(v6 + 32) )
    {
      result = *(unsigned __int8 *)(a1 + 343);
      v9 = result & 0xFFFFFFFFFFFFFFF8LL | (v6 + 32) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v9 )
      {
        result = *(_QWORD *)(v9 + 24);
        for ( i = *(_QWORD **)(v9 + 32); (_QWORD *)result != i; *(_BYTE *)(v11 + 12) |= 0x80u )
        {
          v11 = *(_QWORD *)result;
          result += 8LL;
        }
      }
    }
  }
  return result;
}
