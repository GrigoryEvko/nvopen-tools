// Function: sub_736EA0
// Address: 0x736ea0
//
__int64 __fastcall sub_736EA0(__int64 a1, unsigned int a2)
{
  __int64 v2; // kr28_8
  __int64 result; // rax
  _QWORD *v4; // rbx
  __int64 i; // rbx
  unsigned int v6; // eax
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rdx

  while ( 2 )
  {
    LODWORD(result) = *(unsigned __int8 *)(a1 + 140) - 6;
    v2 = (unsigned int)result;
    result = (unsigned __int8)result;
    switch ( (char)result )
    {
      case 0:
        v6 = sub_8D3410(*(_QWORD *)(a1 + 160));
        return sub_7370D0(*(_QWORD *)(a1 + 160), v6);
      case 1:
        sub_7370D0(*(_QWORD *)(a1 + 160), 0);
        result = *(_QWORD *)(a1 + 168);
        v4 = *(_QWORD **)result;
        if ( *(_QWORD *)result )
        {
          do
          {
            result = sub_7370D0(v4[1], 0);
            v4 = (_QWORD *)*v4;
          }
          while ( v4 );
        }
        return result;
      case 2:
        return sub_7370D0(*(_QWORD *)(a1 + 160), a2);
      case 3:
        return result;
      case 4:
      case 5:
        if ( a2 )
        {
          result = sub_736DD0(a1);
          if ( !(_DWORD)result )
          {
            for ( i = *(_QWORD *)(a1 + 160); i; i = *(_QWORD *)(i + 112) )
              result = sub_7370D0(*(_QWORD *)(i + 120), a2);
          }
        }
        return result;
      case 6:
        if ( !a2 && *(_QWORD *)(a1 + 8) && (unsigned int)sub_8D3410(a1) && (*(_BYTE *)(a1 + 144) & 4) == 0 )
        {
          result = *(unsigned __int8 *)(a1 + 144);
          if ( (result & 2) == 0 )
            return result;
          *(_BYTE *)(a1 + 144) = result | 4;
          v7 = *(_BYTE *)(a1 + 140) == 12;
          *(_BYTE *)(a1 + 144) = result | 0xC;
          if ( !v7
            || !*(_QWORD *)(a1 + 8)
            || (a2 = 1, (unsigned __int8)(*(_BYTE *)(sub_8D21F0(*(_QWORD *)(a1 + 160)) + 140) - 9) > 2u) )
          {
            sub_736EA0(a1, 1);
            result = qword_4F07A30;
            v8 = qword_4F07A30 + 8;
            *(_QWORD *)qword_4F07A30 = a1;
            qword_4F07A30 = v8;
            return result;
          }
          v9 = qword_4F07A30 + 8;
          *(_QWORD *)qword_4F07A30 = a1;
          qword_4F07A30 = v9;
          continue;
        }
        result = sub_736DD0(a1);
        if ( !(_DWORD)result )
          return sub_7370D0(*(_QWORD *)(a1 + 160), a2);
        return result;
      default:
        return v2;
    }
  }
}
