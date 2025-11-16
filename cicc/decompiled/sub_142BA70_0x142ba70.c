// Function: sub_142BA70
// Address: 0x142ba70
//
__int64 __fastcall sub_142BA70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  __int64 i; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // [rsp+0h] [rbp-D0h] BYREF
  _BYTE v12[192]; // [rsp+10h] [rbp-C0h] BYREF

  sub_16C1840(v12);
  sub_16C1A90(v12, a2, a3);
  sub_16C1AA0(v12, &v11);
  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    v5 = a1 + 8;
    do
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(result + 16);
        v7 = *(_QWORD *)(result + 24);
        if ( v11 <= *(_QWORD *)(result + 32) )
          break;
        result = *(_QWORD *)(result + 24);
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = result;
      result = *(_QWORD *)(result + 16);
    }
    while ( v6 );
LABEL_6:
    if ( a1 + 8 != v5 && v11 >= *(_QWORD *)(v5 + 32) )
    {
      result = 4LL * *(unsigned __int8 *)(a1 + 178);
      v8 = result & 0xFFFFFFFFFFFFFFF8LL | (v5 + 32) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 )
      {
        result = *(_QWORD *)(v8 + 24);
        for ( i = *(_QWORD *)(v8 + 32); result != i; *(_BYTE *)(v10 + 12) |= 0x20u )
        {
          v10 = *(_QWORD *)result;
          result += 8;
        }
      }
    }
  }
  return result;
}
