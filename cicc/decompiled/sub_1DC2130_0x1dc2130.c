// Function: sub_1DC2130
// Address: 0x1dc2130
//
__int64 __fastcall sub_1DC2130(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 i; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 result; // rax
  __int64 v10; // r12
  char v11; // al
  __int64 v12; // r14
  signed int v13; // esi

  for ( i = a2; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v7 = *(_QWORD *)(a2 + 24) + 24LL;
  while ( 1 )
  {
    v8 = *(_QWORD *)(i + 32);
    result = 5LL * *(unsigned int *)(i + 40);
    v10 = v8 + 40LL * *(unsigned int *)(i + 40);
    if ( v8 != v10 )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v7 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
      goto LABEL_18;
  }
  do
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)v8 )
      {
        v11 = *(_BYTE *)(v8 + 4);
        if ( (v11 & 1) == 0
          && (v11 & 2) == 0
          && ((*(_BYTE *)(v8 + 3) & 0x10) == 0 || (*(_DWORD *)v8 & 0xFFF00) != 0)
          && (v11 & 8) == 0 )
        {
          v13 = *(_DWORD *)(v8 + 8);
          if ( v13 > 0 )
          {
            v12 = v8 + 40;
            sub_1DC1BF0(a1, v13, a3, a4, a5, a6);
            result = v10;
            if ( v12 == v10 )
              break;
            goto LABEL_23;
          }
        }
      }
      v12 = v8 + 40;
      result = v10;
      if ( v12 == v10 )
        break;
LABEL_23:
      v10 = v12;
LABEL_24:
      v8 = v10;
      v10 = result;
    }
    while ( 1 )
    {
      i = *(_QWORD *)(i + 8);
      if ( v7 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
        break;
      v10 = *(_QWORD *)(i + 32);
      result = v10 + 40LL * *(unsigned int *)(i + 40);
      if ( v10 != result )
        goto LABEL_24;
    }
    v8 = v10;
    v10 = result;
LABEL_18:
    ;
  }
  while ( v8 != v10 );
  return result;
}
