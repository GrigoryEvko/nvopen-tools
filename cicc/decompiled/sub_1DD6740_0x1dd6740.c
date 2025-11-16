// Function: sub_1DD6740
// Address: 0x1dd6740
//
void __fastcall sub_1DD6740(__int64 a1)
{
  char *v1; // r13
  char *v2; // r14
  unsigned __int64 v4; // rax
  __int16 *v5; // rdx
  __int16 *v6; // rsi
  __int16 *v7; // r8
  __int16 *v8; // rax
  __int16 v9; // di
  int v10; // ecx
  char *i; // rsi
  unsigned __int16 v12; // cx
  int v13; // edi
  char *v14; // rdx
  char *j; // rax

  v1 = *(char **)(a1 + 160);
  v2 = *(char **)(a1 + 152);
  if ( v1 != v2 )
  {
    _BitScanReverse64(&v4, (v1 - v2) >> 3);
    sub_1DD50F0(*(_QWORD *)(a1 + 152), *(_QWORD *)(a1 + 160), 2LL * (int)(63 - (v4 ^ 0x3F)));
    if ( v1 - v2 > 128 )
    {
      sub_1DD4EB0(v2, v2 + 128);
      for ( i = v2 + 128; v1 != i; *((_DWORD *)v14 + 1) = v13 )
      {
        v12 = *(_WORD *)i;
        v13 = *((_DWORD *)i + 1);
        v14 = i;
        for ( j = i - 8; v12 < *(_WORD *)j; j -= 8 )
        {
          *((_QWORD *)j + 1) = *(_QWORD *)j;
          v14 = j;
        }
        i += 8;
        *(_WORD *)v14 = v12;
      }
    }
    else
    {
      sub_1DD4EB0(v2, v1);
    }
    v5 = *(__int16 **)(a1 + 152);
    v6 = *(__int16 **)(a1 + 160);
    if ( v6 != v5 )
    {
      v7 = *(__int16 **)(a1 + 152);
      do
      {
        v8 = v5 + 4;
        v9 = *v5;
        v10 = *((_DWORD *)v5 + 1);
        if ( v5 + 4 == v6 )
        {
LABEL_19:
          v5 = v6;
        }
        else
        {
          while ( 1 )
          {
            v5 = v8;
            if ( v9 != *v8 )
              break;
            v10 |= *((_DWORD *)v8 + 1);
            v8 += 4;
            if ( v6 == v8 )
              goto LABEL_19;
          }
        }
        *v7 = v9;
        v7 += 4;
        *((_DWORD *)v7 - 1) = v10;
        v6 = *(__int16 **)(a1 + 160);
      }
      while ( v6 != v5 );
      if ( v7 != v5 )
        *(_QWORD *)(a1 + 160) = v7;
    }
  }
}
