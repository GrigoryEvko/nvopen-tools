// Function: sub_7C9B80
// Address: 0x7c9b80
//
void __fastcall sub_7C9B80(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  __int64 v6; // rbx
  __int16 v7; // ax
  __int64 v8; // rdx
  __int64 v9; // rax

  v5 = a3;
  v6 = a1;
  if ( a2 )
  {
    while ( v6 )
    {
      if ( *(_DWORD *)(v6 + 28) >= a2 )
        goto LABEL_7;
      v6 = *(_QWORD *)v6;
    }
  }
  else if ( a1 )
  {
LABEL_7:
    while ( 1 )
    {
      v7 = *(_WORD *)(v6 + 24);
      if ( v7 == 9 || v5 && *(_DWORD *)(v6 + 28) >= v5 )
        break;
      if ( v7 == 16 )
        v6 = *(_QWORD *)(v6 + 64);
      if ( *(_BYTE *)(v6 + 26) == 5 )
      {
        if ( *(_QWORD *)(v6 + 64) )
        {
          v6 = *(_QWORD *)(v6 + 64);
          sub_729660(59);
        }
        else
        {
          v8 = *(_QWORD *)(v6 + 48);
          switch ( *(_BYTE *)(v8 + 80) )
          {
            case 4:
            case 5:
              v9 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 80LL);
              break;
            case 6:
              v9 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v9 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v9 = *(_QWORD *)(v8 + 88);
              break;
            default:
              BUG();
          }
          sub_7C9B80(*(_QWORD *)(v9 + 8), 0, 0);
          if ( *(_WORD *)(v6 + 24) != 17 && !*(_BYTE *)(v6 + 56) )
            sub_7AD2D0(v6, 0, a3, a4, a5);
        }
        v6 = *(_QWORD *)v6;
        if ( !v6 )
          return;
      }
      else
      {
        sub_7C9740(v6, 0, a3, a4, a5);
        v6 = *(_QWORD *)v6;
        if ( !v6 )
          return;
      }
    }
  }
}
