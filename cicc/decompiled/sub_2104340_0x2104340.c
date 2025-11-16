// Function: sub_2104340
// Address: 0x2104340
//
__int64 __fastcall sub_2104340(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 i; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // r12
  int v7; // edx
  char v8; // al
  __int64 v9; // rcx
  _WORD *v10; // r8
  _WORD *v11; // rsi
  unsigned __int64 v12; // rcx
  _WORD *v13; // rdx
  _QWORD *v14; // rax
  int v15; // eax
  __int64 v16; // r14
  __int64 v17; // rsi

  for ( i = a2; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v3 = *(_QWORD *)(a2 + 24) + 24LL;
  do
  {
    v4 = *(_QWORD *)(i + 32);
    result = 5LL * *(unsigned int *)(i + 40);
    v6 = v4 + 40LL * *(unsigned int *)(i + 40);
    if ( v4 != v6 )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v3 == i )
      break;
  }
  while ( (*(_BYTE *)(i + 46) & 4) != 0 );
  if ( v4 != v6 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v4 )
      {
        if ( *(_BYTE *)v4 == 12 )
        {
          v17 = *(_QWORD *)(v4 + 24);
          v16 = v4 + 40;
          sub_2103F30(a1, v17);
          result = v6;
          if ( v16 == v6 )
            goto LABEL_22;
          goto LABEL_27;
        }
      }
      else
      {
        v7 = *(_DWORD *)(v4 + 8);
        if ( v7 > 0 )
        {
          if ( (*(_BYTE *)(v4 + 3) & 0x10) != 0 || (v8 = *(_BYTE *)(v4 + 4), (v8 & 1) == 0) && (v8 & 2) == 0 )
          {
            v9 = *a1;
            if ( !*a1 )
              BUG();
            v10 = (_WORD *)(*(_QWORD *)(v9 + 56)
                          + 2LL * (*(_DWORD *)(*(_QWORD *)(v9 + 8) + 24LL * (unsigned int)v7 + 16) >> 4));
            v12 = v7 * (*(_DWORD *)(*(_QWORD *)(v9 + 8) + 24LL * (unsigned int)v7 + 16) & 0xFu);
            v11 = v10 + 1;
            LOWORD(v12) = *v10 + v12;
            while ( 1 )
            {
              v13 = v11;
              if ( !v11 )
                break;
              while ( 1 )
              {
                ++v13;
                v14 = (_QWORD *)(a1[1] + ((v12 >> 3) & 0x1FF8));
                *v14 |= 1LL << v12;
                v15 = (unsigned __int16)*(v13 - 1);
                v11 = 0;
                if ( !(_WORD)v15 )
                  break;
                v12 = (unsigned int)(v15 + v12);
                if ( !v13 )
                  goto LABEL_18;
              }
            }
          }
        }
      }
LABEL_18:
      v16 = v4 + 40;
      result = v6;
      if ( v16 == v6 )
      {
LABEL_22:
        while ( 1 )
        {
          i = *(_QWORD *)(i + 8);
          if ( v3 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
            break;
          v6 = *(_QWORD *)(i + 32);
          result = v6 + 40LL * *(unsigned int *)(i + 40);
          if ( v6 != result )
            goto LABEL_28;
        }
        v4 = v6;
        v6 = result;
        if ( v4 == result )
          return result;
      }
      else
      {
LABEL_27:
        v6 = v16;
LABEL_28:
        v4 = v6;
        v6 = result;
      }
    }
  }
  return result;
}
