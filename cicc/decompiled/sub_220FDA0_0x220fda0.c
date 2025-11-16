// Function: sub_220FDA0
// Address: 0x220fda0
//
__int64 __fastcall sub_220FDA0(__int64 a1, unsigned int a2)
{
  _BYTE *v2; // rax
  __int64 result; // rax
  _BYTE *v4; // rcx
  _BYTE *v5; // rax
  _BYTE *v6; // rcx
  _BYTE *v7; // rdx
  _BYTE *v8; // rdx
  _BYTE *v9; // rcx

  if ( a2 <= 0x7F )
  {
    v2 = *(_BYTE **)a1;
    if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a1 )
      return 0;
    *(_QWORD *)a1 = v2 + 1;
    *v2 = a2;
    return 1;
  }
  if ( a2 <= 0x7FF )
  {
    v4 = *(_BYTE **)a1;
    result = 0;
    if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 <= 1u )
      return result;
    *(_QWORD *)a1 = v4 + 1;
    *v4 = (a2 >> 6) - 64;
LABEL_7:
    v5 = (_BYTE *)(*(_QWORD *)a1)++;
    *v5 = (a2 & 0x3F) + 0x80;
    return 1;
  }
  if ( a2 <= 0xFFFF )
  {
    v9 = *(_BYTE **)a1;
    result = 0;
    if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 <= 2u )
      return result;
    *(_QWORD *)a1 = v9 + 1;
    *v9 = (a2 >> 12) - 32;
    goto LABEL_12;
  }
  result = 0;
  if ( a2 <= 0x10FFFF )
  {
    v6 = *(_BYTE **)a1;
    if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 > 3u )
    {
      *(_QWORD *)a1 = v6 + 1;
      *v6 = (a2 >> 18) - 16;
      v7 = (_BYTE *)(*(_QWORD *)a1)++;
      *v7 = ((a2 >> 12) & 0x3F) + 0x80;
LABEL_12:
      v8 = (_BYTE *)(*(_QWORD *)a1)++;
      *v8 = ((a2 >> 6) & 0x3F) + 0x80;
      goto LABEL_7;
    }
  }
  return result;
}
