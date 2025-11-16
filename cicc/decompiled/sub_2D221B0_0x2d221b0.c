// Function: sub_2D221B0
// Address: 0x2d221b0
//
__int64 __fastcall sub_2D221B0(unsigned int **a1, __int64 a2)
{
  __int64 *v3; // rax
  struct __jmp_buf_tag *v4; // r12
  __int64 *v5; // r13
  _BYTE *v6; // rax
  __int64 i; // r13
  __int64 j; // r15
  __int64 v9; // r8
  int v10; // esi
  unsigned int v11; // edi
  unsigned __int8 v12; // al
  __int64 *v13; // rax
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  if ( (unsigned __int8)sub_2D21D20(a1, a2) )
  {
    v3 = sub_CEACC0();
    v4 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v3);
    if ( v4 )
    {
      v5 = sub_CEAD60();
      v6 = (_BYTE *)sub_CEECD0(1, 1u);
      *v6 = 1;
      sub_C94E10((__int64)v5, v6);
      longjmp(v4, 1);
    }
  }
  if ( unk_50165C8 )
  {
    v14 = a2 + 24;
    v16 = *(_QWORD *)(a2 + 32);
    if ( v16 != a2 + 24 )
    {
      do
      {
        if ( !v16 )
          BUG();
        for ( i = *(_QWORD *)(v16 + 24); v16 + 16 != i; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          for ( j = *(_QWORD *)(i + 32); i + 24 != j; j = *(_QWORD *)(j + 8) )
          {
            if ( !j )
              BUG();
            if ( *(_BYTE *)(j - 24) > 0x1Cu )
            {
              switch ( *(_BYTE *)(j - 24) )
              {
                case ')':
                case '+':
                case '-':
                case '/':
                case '2':
                case '5':
                case 'J':
                case 'K':
                case 'S':
                  goto LABEL_28;
                case 'T':
                case 'U':
                case 'V':
                  v9 = *(_QWORD *)(j - 16);
                  v10 = *(unsigned __int8 *)(v9 + 8);
                  v11 = v10 - 17;
                  v12 = *(_BYTE *)(v9 + 8);
                  if ( (unsigned int)(v10 - 17) <= 1 )
                    v12 = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
                  if ( v12 <= 3u || v12 == 5 || (v12 & 0xFD) == 4 )
                    goto LABEL_28;
                  if ( (_BYTE)v10 == 15 )
                  {
                    if ( (*(_BYTE *)(v9 + 9) & 4) == 0 )
                      continue;
                    v15 = *(_QWORD *)(j - 16);
                    if ( !sub_BCB420(v15) )
                      continue;
                    v13 = *(__int64 **)(v15 + 16);
                    v9 = *v13;
                    v10 = *(unsigned __int8 *)(*v13 + 8);
                    v11 = v10 - 17;
                  }
                  else if ( (_BYTE)v10 == 16 )
                  {
                    do
                    {
                      v9 = *(_QWORD *)(v9 + 24);
                      LOBYTE(v10) = *(_BYTE *)(v9 + 8);
                    }
                    while ( (_BYTE)v10 == 16 );
                    v11 = (unsigned __int8)v10 - 17;
                  }
                  if ( v11 <= 1 )
                    LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
                  if ( (unsigned __int8)v10 <= 3u || (_BYTE)v10 == 5 || (v10 & 0xFD) == 4 )
LABEL_28:
                    sub_B44E70(j - 24, 1);
                  break;
                default:
                  continue;
              }
            }
          }
        }
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v14 != v16 );
    }
  }
  return 0;
}
