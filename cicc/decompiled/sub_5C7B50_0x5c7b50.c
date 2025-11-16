// Function: sub_5C7B50
// Address: 0x5c7b50
//
__int64 __fastcall sub_5C7B50(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r12
  __int64 v4; // r13
  char *v6; // rax
  char *v7; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a2;
  switch ( a3 )
  {
    case 3:
      v4 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
      v3 = *(_QWORD *)a2 + 8LL;
      break;
    case 6:
      v4 = *(_QWORD *)a2;
      if ( (*(_BYTE *)(a1 + 11) & 2) == 0 && *(_BYTE *)(v4 + 140) == 12 && *(_QWORD *)(v4 + 8) )
      {
        v3 = v4 + 160;
        v4 = *(_QWORD *)(v4 + 160);
      }
      break;
    case 7:
    case 8:
      v4 = *(_QWORD *)(*(_QWORD *)a2 + 120LL);
      v3 = *(_QWORD *)a2 + 120LL;
      break;
    case 11:
      v4 = *(_QWORD *)(*(_QWORD *)a2 + 152LL);
      v3 = *(_QWORD *)a2 + 152LL;
      break;
    default:
      sub_721090(a1);
  }
  if ( (unsigned int)sub_8D2310(v4) || (unsigned int)sub_8D2340(v4) )
  {
    sub_73EA10(v3, v8);
    return v8[0];
  }
  else if ( (unsigned int)sub_8DBE70(v4) )
  {
    return 0;
  }
  else
  {
    if ( sub_8D49A0(v4) )
    {
      v6 = sub_5C79F0(a1);
      sub_686450(1142, a1 + 56, v6, v4);
    }
    else
    {
      v7 = sub_5C79F0(a1);
      sub_684B10(2527, a1 + 56, v7);
    }
    *(_BYTE *)(a1 + 8) = 0;
    return 0;
  }
}
