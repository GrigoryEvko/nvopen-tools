// Function: sub_890050
// Address: 0x890050
//
void __fastcall sub_890050(__int64 a1, __int64 a2)
{
  __int64 i; // rdx
  __int64 v4; // rax
  _QWORD *j; // rbx
  __int64 v6; // rdi
  __int64 v7; // rdx

  for ( i = *(_QWORD *)a2; (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a2 + 80LL) - 4) <= 1u; i = *(_QWORD *)a2 )
  {
    v4 = *(_QWORD *)(i + 88);
    if ( !v4 || (*(_BYTE *)(v4 + 177) & 0x20) != 0 )
      break;
    for ( j = *(_QWORD **)(a1 + 72); j; j = (_QWORD *)*j )
    {
      v6 = j[1];
      if ( v6 != a2 )
        sub_5ED880(v6, a2, 1, 0);
    }
    v7 = *(_QWORD *)(a1 + 88);
    if ( !v7 )
      break;
    switch ( *(_BYTE *)(v7 + 80) )
    {
      case 4:
      case 5:
        a1 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 80LL);
        break;
      case 6:
        a1 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        a1 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        a1 = *(_QWORD *)(v7 + 88);
        break;
      default:
        a1 = 0;
        break;
    }
  }
}
