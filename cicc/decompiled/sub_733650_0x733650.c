// Function: sub_733650
// Address: 0x733650
//
void __fastcall sub_733650(__int64 a1)
{
  _QWORD *v1; // rdx
  _QWORD *v2; // rdx
  _QWORD *v3; // rax

  if ( *(_BYTE *)a1 == 2 )
  {
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  else
  {
    v1 = *(_QWORD **)(a1 + 16);
    switch ( *(_BYTE *)(a1 + 8) )
    {
      case 0xD:
        v2 = v1 + 8;
        break;
      case 0x13:
        v2 = v1 + 3;
        break;
      case 0x14:
        v2 = v1 + 2;
        break;
      case 0x17:
        v3 = v1 + 7;
        v2 = v1 + 11;
        if ( *(_BYTE *)a1 == 3 )
          v2 = v3;
        break;
      case 0x1E:
        v2 = v1 + 5;
        break;
      case 0x1F:
        v2 = v1 + 4;
        break;
      default:
        sub_721090();
    }
    *v2 = 0;
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
}
