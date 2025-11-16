// Function: sub_867A10
// Address: 0x867a10
//
__int64 sub_867A10()
{
  __int64 v0; // rax
  __int64 v1; // rax

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( v0 )
  {
    while ( 2 )
    {
      switch ( *(_BYTE *)(v0 + 4) )
      {
        case 0:
        case 3:
        case 4:
          return 0;
        case 6:
        case 7:
        case 0x10:
          if ( *(char *)(*(_QWORD *)(v0 + 208) + 90LL) < 0 )
            return 1;
          goto LABEL_4;
        case 0x11:
          if ( *(char *)(*(_QWORD *)(v0 + 216) + 90LL) >= 0 )
            goto LABEL_4;
          return 1;
        default:
LABEL_4:
          v1 = *(int *)(v0 + 552);
          if ( (_DWORD)v1 == -1 )
            return 0;
          v0 = qword_4F04C68[0] + 776 * v1;
          if ( !v0 )
            return 0;
          continue;
      }
    }
  }
  return 0;
}
