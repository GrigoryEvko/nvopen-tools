// Function: sub_2BFE150
// Address: 0x2bfe150
//
__int64 __fastcall sub_2BFE150(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // [rsp+18h] [rbp-28h] BYREF
  __int64 v7[4]; // [rsp+20h] [rbp-20h] BYREF

  v2 = *(_BYTE *)(a2 + 160);
  v7[0] = a1;
  v7[1] = a2;
  if ( (unsigned __int8)(v2 - 12) > 0x12u )
  {
    switch ( v2 )
    {
      case '5':
      case 'I':
      case 'S':
        return sub_BCCE00(*(_QWORD **)(a1 + 40), 1u);
      case '9':
        v3 = sub_2BFD6A0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        v6 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL);
        *sub_2BFD200(a1, &v6) = v3;
        return v3;
      case 'E':
      case 'F':
      case 'K':
      case 'L':
      case 'M':
      case 'U':
        return sub_2BFDEB0(v7);
      case 'J':
        return sub_BCD140(*(_QWORD **)(a1 + 40), 0x20u);
      case 'N':
      case 'O':
        return sub_BCB120(*(_QWORD **)(a1 + 40));
      case 'P':
      case 'T':
        return sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
      case 'Q':
        v5 = **(_QWORD **)(a2 + 48);
        if ( !v5 )
          BUG();
        return *(_QWORD *)(*(_QWORD *)(v5 + 40) + 8LL);
      case 'R':
      case 'V':
        v3 = sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
        if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
          return *(_QWORD *)(v3 + 24);
        return v3;
      default:
        BUG();
    }
  }
  return sub_2BFDEB0(v7);
}
