// Function: sub_5E71C0
// Address: 0x5e71c0
//
__int64 __fastcall sub_5E71C0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  _QWORD *v4; // r12
  char v5; // r14
  __int64 v6; // rax

  while ( 2 )
  {
    while ( 1 )
    {
      result = *(unsigned __int8 *)(a1 + 140);
      if ( (_BYTE)result != 12 )
        break;
      a1 = *(_QWORD *)(a1 + 160);
    }
    switch ( (char)result )
    {
      case 2:
        if ( (*(_BYTE *)(a1 + 161) & 8) == 0 )
          return result;
        if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
          goto LABEL_17;
        result = sub_5E7150(a1);
        if ( (_DWORD)result )
        {
          result = *(_BYTE *)(a1 + 88) & 0x8F | 0x20u;
          *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x20;
          ++*a2;
        }
        return result;
      case 6:
        a1 = sub_8D46C0(a1);
        continue;
      case 7:
        sub_5E71C0(*(_QWORD *)(a1 + 160), a2);
        result = *(_QWORD *)(a1 + 168);
        v4 = *(_QWORD **)result;
        if ( *(_QWORD *)result )
        {
          do
          {
            result = sub_5E71C0(v4[1], a2);
            v4 = (_QWORD *)*v4;
          }
          while ( v4 );
        }
        return result;
      case 8:
        a1 = sub_8D4050(a1);
        continue;
      case 9:
      case 10:
      case 11:
        if ( !(unsigned int)sub_5E7150(a1) )
        {
          result = *(_QWORD *)(a1 + 168);
          if ( *(_QWORD *)(result + 120)
            && (result = *(unsigned __int8 *)(a1 + 88), v5 = (*(_BYTE *)(a1 + 88) >> 4) & 7, v5 != 2) )
          {
            *(_BYTE *)(a1 + 88) = result & 0x8F | 0x20;
            sub_5E7470(a1, a2);
            result = *(_BYTE *)(a1 + 88) & 0x8F;
            *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | (16 * v5);
            if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
              return result;
          }
          else if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
          {
            return result;
          }
LABEL_17:
          a1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
          continue;
        }
        *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x20;
        result = sub_5E7470(a1, a2);
        if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
          goto LABEL_17;
        return result;
      case 13:
        v6 = sub_8D4890(a1);
        sub_5E71C0(v6, a2);
        a1 = sub_8D4870(a1);
        continue;
      default:
        return result;
    }
  }
}
