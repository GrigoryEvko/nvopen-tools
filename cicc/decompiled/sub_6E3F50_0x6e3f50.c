// Function: sub_6E3F50
// Address: 0x6e3f50
//
__int64 __fastcall sub_6E3F50(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rax

  while ( 1 )
  {
    switch ( *(_BYTE *)(a1 + 48) )
    {
      case 0:
      case 1:
        return 0;
      case 2:
        return *(_QWORD *)(*(_QWORD *)(a1 + 56) + 144LL);
      case 3:
      case 4:
        return *(_QWORD *)(a1 + 56);
      case 5:
        return *(_QWORD *)(a1 + 64);
      case 6:
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 176LL);
        if ( *(_BYTE *)(v2 + 173) != 11 )
          goto LABEL_10;
        v3 = *(_QWORD *)(v2 + 176);
        if ( *(_BYTE *)(v3 + 173) != 9 )
          goto LABEL_10;
        a1 = *(_QWORD *)(v3 + 176);
        break;
      case 7:
        result = *(_QWORD *)(a1 + 56);
        if ( !result )
          goto LABEL_10;
        return result;
      default:
LABEL_10:
        sub_721090(a1);
    }
  }
}
