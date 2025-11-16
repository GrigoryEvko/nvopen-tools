// Function: sub_735280
// Address: 0x735280
//
void __fastcall sub_735280(__int64 a1)
{
  __int64 *i; // rbx
  char v3; // al
  __int64 j; // rbx
  char v5; // al
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rax

  for ( i = *(__int64 **)(a1 + 144); i; i = (__int64 *)i[14] )
  {
    while ( 1 )
    {
      if ( (i[11] & 8) == 0 )
      {
        v3 = *((_BYTE *)i + 195);
        if ( (v3 & 8) == 0 && ((v3 & 3) == 1 || dword_4F068EC && *((char *)i + 192) < 0) )
        {
          v6 = *i;
          if ( *i )
          {
            if ( (i[24] & 2) == 0 )
              break;
            v7 = *(_QWORD *)(v6 + 64);
            v8 = *(_QWORD *)(*(_QWORD *)(v7 + 168) + 192LL);
            if ( !v8 || *(char *)(v8 - 8) >= 0 )
            {
              v9 = *(_QWORD *)(v7 + 152);
              if ( !v9 || *(char *)(v9 - 8) >= 0 || unk_4D03F88 && v8 && (*(_BYTE *)(v8 + 174) & 1) != 0 )
                break;
            }
          }
        }
      }
LABEL_7:
      i = (__int64 *)i[14];
      if ( !i )
        goto LABEL_8;
    }
    if ( (unsigned __int8)(*((_BYTE *)i + 174) - 1) <= 1u )
    {
      v10 = (_QWORD *)i[22];
      if ( v10 )
      {
        while ( *(char *)(v10[1] - 8LL) >= 0 )
        {
          v10 = (_QWORD *)*v10;
          if ( !v10 )
            goto LABEL_27;
        }
        goto LABEL_7;
      }
    }
LABEL_27:
    sub_8AD0D0(v6, 0, 2);
  }
LABEL_8:
  for ( j = *(_QWORD *)(a1 + 112); j; j = *(_QWORD *)(j + 112) )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(j + 88) & 8) == 0 )
      {
        v5 = *(_BYTE *)(j + 170);
        if ( (v5 & 0x40) == 0 && (v5 & 0xB0) == 0x10 && *(_QWORD *)j )
          break;
      }
      j = *(_QWORD *)(j + 112);
      if ( !j )
        return;
    }
    sub_8AD0D0(*(_QWORD *)j, 0, 2);
  }
}
