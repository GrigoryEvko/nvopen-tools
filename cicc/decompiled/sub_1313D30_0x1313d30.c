// Function: sub_1313D30
// Address: 0x1313d30
//
__int64 __fastcall sub_1313D30(__int64 a1, char a2)
{
  char v2; // al
  unsigned __int64 v4; // r8
  char v5; // al
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // r8

  v2 = *(_BYTE *)(a1 + 816);
  if ( v2 == 1 )
    return a1;
  if ( v2 == 2 )
  {
    sub_1313A40((_BYTE *)a1);
    return a1;
  }
  else
  {
    switch ( v2 )
    {
      case 6:
        if ( a2 )
        {
          sub_1313AB0(a1, 3u);
          v6 = __readfsqword(0);
          if ( a1 != v6 - 2664 )
            qmemcpy((void *)(v6 - 2664), (const void *)a1, 0xA48u);
          if ( !pthread_setspecific(key, (const void *)(v6 - 2664))
            || (sub_130AA40("<jemalloc>: Error setting tsd.\n"), !byte_4F969A5[0]) )
          {
            sub_130D500((_QWORD *)(a1 + 432));
            *(_QWORD *)(a1 + 112) = a1;
            *(_WORD *)a1 = 256;
            sub_1313840((_QWORD *)a1);
            sub_130DBC0(a1);
            *(_BYTE *)(a1 + 2) = 1;
            return a1;
          }
          goto LABEL_30;
        }
        if ( !unk_4F96B58 )
          return a1;
        sub_1313AB0(a1, 0);
        sub_1313A40((_BYTE *)a1);
        v4 = __readfsqword(0);
        if ( a1 != v4 - 2664 )
          qmemcpy((void *)(v4 - 2664), (const void *)a1, 0xA48u);
        if ( pthread_setspecific(key, (const void *)(v4 - 2664)) )
        {
          sub_130AA40("<jemalloc>: Error setting tsd.\n");
          if ( byte_4F969A5[0] )
LABEL_30:
            abort();
        }
        break;
      case 3:
        v5 = *(_BYTE *)(a1 + 2) + 1;
        *(_BYTE *)(a1 + 2) = v5;
        if ( v5 != (char)0x80 && a2 == 1 )
          return a1;
        sub_1313AB0(a1, 0);
        --*(_BYTE *)(a1 + 1);
        sub_1313A40((_BYTE *)a1);
        break;
      case 4:
        sub_1313AB0(a1, 5u);
        v7 = __readfsqword(0);
        if ( a1 != v7 - 2664 )
          qmemcpy((void *)(v7 - 2664), (const void *)a1, 0xA48u);
        if ( !pthread_setspecific(key, (const void *)(v7 - 2664))
          || (sub_130AA40("<jemalloc>: Error setting tsd.\n"), !byte_4F969A5[0]) )
        {
          sub_130D500((_QWORD *)(a1 + 432));
          *(_QWORD *)(a1 + 112) = a1;
          *(_WORD *)a1 = 256;
          sub_1313840((_QWORD *)a1);
          sub_130DBC0(a1);
          return a1;
        }
        goto LABEL_30;
      default:
        return a1;
    }
    sub_130D500((_QWORD *)(a1 + 432));
    *(_QWORD *)(a1 + 112) = a1;
    sub_1313840((_QWORD *)a1);
    sub_130DBC0(a1);
    sub_1312C30((_BYTE *)a1);
    return a1;
  }
}
