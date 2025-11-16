// Function: sub_7AB890
// Address: 0x7ab890
//
__int64 __fastcall sub_7AB890(unsigned __int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 *v10; // rax

  result = a2;
  if ( a1 > 0x2069 )
  {
    if ( a1 != 65279 )
      return 0;
  }
  else if ( a1 <= 0x2029 )
  {
    if ( a1 - 8203 >= 3 )
      return 0;
  }
  else
  {
    switch ( a1 )
    {
      case 0x202AuLL:
      case 0x202BuLL:
      case 0x202DuLL:
      case 0x202EuLL:
      case 0x2066uLL:
      case 0x2067uLL:
      case 0x2068uLL:
        v3 = (__int64 *)qword_4F084C8;
        if ( qword_4F084C8 )
          qword_4F084C8 = *(_QWORD *)qword_4F084C8;
        else
          v3 = (__int64 *)sub_823970(16);
        v4 = qword_4F084D0;
        v3[1] = a1;
        qword_4F084D0 = (__int64)v3;
        *v3 = v4;
        result = 0;
        break;
      case 0x202CuLL:
        v5 = (__int64 *)qword_4F084D0;
        result = 1;
        if ( qword_4F084D0 )
        {
          v8 = *(_QWORD *)(qword_4F084D0 + 8);
          v6 = qword_4F084C8;
          if ( (unsigned __int64)(v8 - 8234) > 1 && (unsigned __int64)(v8 - 8237) > 1 )
          {
            while ( 1 )
            {
              v9 = (__int64 *)*v5;
              *v5 = v6;
              v6 = (__int64)v5;
              if ( !v9 )
                break;
              v5 = v9;
            }
            goto LABEL_24;
          }
          goto LABEL_17;
        }
        break;
      case 0x2060uLL:
        return result;
      case 0x2069uLL:
        v5 = (__int64 *)qword_4F084D0;
        result = 1;
        if ( qword_4F084D0 )
        {
          v6 = qword_4F084C8;
          if ( (unsigned __int64)(*(_QWORD *)(qword_4F084D0 + 8) - 8294LL) > 2 )
          {
            while ( 1 )
            {
              v10 = (__int64 *)*v5;
              *v5 = v6;
              v6 = (__int64)v5;
              if ( !v10 )
                break;
              v5 = v10;
            }
LABEL_24:
            qword_4F084C8 = (__int64)v5;
            qword_4F084D0 = 0;
            result = 1;
          }
          else
          {
LABEL_17:
            v7 = *v5;
            *v5 = v6;
            qword_4F084C8 = (__int64)v5;
            qword_4F084D0 = v7;
            result = 0;
          }
        }
        break;
      default:
        result = 0;
        break;
    }
  }
  return result;
}
