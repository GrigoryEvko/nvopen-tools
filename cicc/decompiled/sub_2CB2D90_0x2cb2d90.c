// Function: sub_2CB2D90
// Address: 0x2cb2d90
//
__int64 __fastcall sub_2CB2D90(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d
  int v3; // eax
  char v4; // al
  char v5; // dl

  v1 = *(_QWORD *)(a1 - 32);
  v2 = 0;
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v3 = *(_DWORD *)(v1 + 36);
  if ( v3 != 173 )
  {
    switch ( v3 )
    {
      case 8654:
        LOBYTE(v2) = 2;
        return v2;
      case 8656:
        LOBYTE(v2) = 18;
        return v2;
      case 8663:
        break;
      case 8669:
        LOBYTE(v2) = 17;
        return v2;
      case 8692:
        v4 = 0;
        v5 = 3;
        goto LABEL_7;
      case 8694:
        LOBYTE(v2) = 19;
        return v2;
      case 8699:
        v4 = 0;
        v5 = 4;
        goto LABEL_7;
      case 8701:
        v4 = 1;
        v5 = 4;
        goto LABEL_7;
      default:
        BUG();
    }
  }
  v4 = 0;
  v5 = 1;
LABEL_7:
  LOBYTE(v2) = v5 | (16 * v4);
  return v2;
}
