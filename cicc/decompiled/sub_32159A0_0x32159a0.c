// Function: sub_32159A0
// Address: 0x32159a0
//
__int64 __fastcall sub_32159A0(__int64 a1, __int64 a2, __int16 a3)
{
  char v3; // al

  switch ( a3 )
  {
    case 16:
      if ( *(_WORD *)a2 == 2 )
        return *(unsigned __int8 *)(a2 + 2);
      v3 = *(_BYTE *)(a2 + 3);
      if ( !v3 )
        return 4;
      if ( v3 != 1 )
LABEL_11:
        BUG();
      return 8;
    case 17:
      return 1;
    case 18:
      return 2;
    case 19:
      return 4;
    case 20:
      return 8;
    case 21:
      return sub_F03EF0(*(unsigned int *)(*(_QWORD *)a1 + 16LL));
    default:
      goto LABEL_11;
  }
}
