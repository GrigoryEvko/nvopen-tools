// Function: sub_876E10
// Address: 0x876e10
//
__int16 __fastcall sub_876E10(__int64 a1, __int64 a2, FILE *a3, unsigned int a4, unsigned int a5, _DWORD *a6)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // edx

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v9 = *(_QWORD *)(v8 + 8);
  if ( v9 )
  {
    LOWORD(v8) = *(unsigned __int8 *)(v9 + 80);
    v10 = 0;
    if ( (_BYTE)v8 == 17 )
    {
      v9 = *(_QWORD *)(v9 + 88);
      if ( !v9 )
        return v8;
      LOWORD(v8) = *(unsigned __int8 *)(v9 + 80);
      v10 = 1;
    }
    LOWORD(v8) = v8 - 10;
    if ( (unsigned __int8)v8 <= 1u )
      goto LABEL_10;
    while ( v10 )
    {
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        break;
      LOWORD(v8) = *(unsigned __int8 *)(v9 + 80) - 10;
      if ( (unsigned __int8)(*(_BYTE *)(v9 + 80) - 10) <= 1u )
      {
LABEL_10:
        v8 = *(_QWORD *)(v9 + 88);
        if ( (*(_BYTE *)(v8 + 194) & 4) != 0 )
        {
          LOWORD(v8) = sub_8769C0(v9, a3, a2, 0, 0, 0, a4, a5, a6);
          return v8;
        }
      }
    }
  }
  return v8;
}
