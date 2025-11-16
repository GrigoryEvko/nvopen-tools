// Function: sub_21BE290
// Address: 0x21be290
//
char __fastcall sub_21BE290(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  switch ( a2 )
  {
    case 0:
      LOBYTE(v2) = sub_21BE280(a1);
      return v2;
    case 1:
      v2 = *(_QWORD *)(a1 + 480);
      if ( *(_DWORD *)(v2 + 248) <= 0x3Bu )
        goto LABEL_17;
      LOBYTE(v2) = *(_DWORD *)(v2 + 252) > 0x1Du;
      return v2;
    case 2:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x1Fu;
      return v2;
    case 3:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) <= 0x1Fu;
      return v2;
    case 4:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 248LL) > 0x1Eu;
      return v2;
    case 5:
      if ( !(unsigned __int8)sub_21BE190(a1) )
        goto LABEL_17;
      goto LABEL_14;
    case 6:
LABEL_14:
      LOBYTE(v2) = sub_21BE150(a1);
      return v2;
    case 7:
      goto LABEL_12;
    case 8:
      v2 = *(_QWORD *)(a1 + 480);
      if ( *(_DWORD *)(v2 + 248) > 0x3Cu )
        goto LABEL_5;
      LOBYTE(v2) = 0;
      return v2;
    case 9:
      v2 = *(_QWORD *)(a1 + 480);
      if ( *(_DWORD *)(v2 + 248) > 0x3Bu )
        goto LABEL_5;
      LOBYTE(v2) = 0;
      return v2;
    case 10:
    case 11:
    case 12:
    case 23:
      v2 = *(_QWORD *)(a1 + 480);
LABEL_5:
      LOBYTE(v2) = *(_DWORD *)(v2 + 252) > 0x45u;
      return v2;
    case 13:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0xAu;
      return v2;
    case 14:
    case 16:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0xBu;
      return v2;
    case 15:
    case 17:
    case 18:
    case 24:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x13u;
      return v2;
    case 19:
    case 21:
    case 22:
    case 25:
    case 26:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x3Bu;
      return v2;
    case 20:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x59u;
      return v2;
    case 27:
      LOBYTE(v2) = *(_BYTE *)(a1 + 472);
      return v2;
    case 28:
      goto LABEL_16;
    case 29:
    case 40:
      goto LABEL_9;
    case 30:
      v3 = *(_QWORD *)(a1 + 480);
      if ( *(_DWORD *)(v3 + 252) <= 0x3Cu )
        goto LABEL_17;
      LOBYTE(v2) = 1;
      if ( *(_DWORD *)(v3 + 248) <= 0x31u )
        goto LABEL_17;
      return v2;
    case 31:
      LOBYTE(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x31u;
      return v2;
    case 32:
      if ( !(unsigned __int8)sub_21BE190(a1) )
        goto LABEL_31;
      LOBYTE(v2) = 0;
      return v2;
    case 33:
      if ( !(unsigned __int8)sub_21BE190(a1) )
        goto LABEL_17;
LABEL_31:
      if ( (unsigned int)sub_21BE110(a1) && (unsigned int)sub_21BE110(a1) != 1 )
        goto LABEL_17;
      LODWORD(v2) = sub_21BE150(a1) ^ 1;
      return v2;
    case 34:
      if ( (unsigned __int8)sub_21BE190(a1) )
        goto LABEL_23;
      goto LABEL_17;
    case 35:
LABEL_23:
      LOBYTE(v2) = (unsigned int)sub_21BE110(a1) == 0;
      return v2;
    case 36:
      if ( (unsigned __int8)sub_21BE190(a1) )
        goto LABEL_21;
      goto LABEL_17;
    case 37:
LABEL_21:
      LOBYTE(v2) = (unsigned int)sub_21BE110(a1) == 1;
      return v2;
    case 38:
      if ( (int)sub_21BE110(a1) <= 2 )
        goto LABEL_12;
      LOBYTE(v2) = 0;
      return v2;
    case 39:
      LOBYTE(v2) = 1;
      return v2;
    case 41:
      LOBYTE(v2) = sub_21BE1E0(a1);
      return v2;
    case 42:
      goto LABEL_19;
    case 43:
      if ( !(unsigned __int8)sub_21BE1E0(a1) )
        goto LABEL_17;
LABEL_12:
      LOBYTE(v2) = sub_21BE190(a1);
      return v2;
    case 44:
      if ( !(unsigned __int8)sub_21BE190(a1) )
        goto LABEL_17;
LABEL_19:
      LODWORD(v2) = sub_21BE1E0(a1) ^ 1;
      return v2;
    case 45:
      if ( !(unsigned __int8)sub_21BE1E0(a1) )
        goto LABEL_17;
LABEL_16:
      if ( !(unsigned __int8)sub_21BE190(a1) )
        goto LABEL_17;
      goto LABEL_9;
    case 46:
      if ( (unsigned __int8)sub_21BE1E0(a1) )
        goto LABEL_9;
      LOBYTE(v2) = 0;
      return v2;
    case 47:
      if ( (unsigned __int8)sub_21BE190(a1) )
        goto LABEL_8;
      goto LABEL_17;
    case 48:
LABEL_8:
      if ( (unsigned __int8)sub_21BE1E0(a1) )
LABEL_17:
        LOBYTE(v2) = 0;
      else
LABEL_9:
        LOBYTE(v2) = sub_21652E0(*(_QWORD *)(a1 + 480));
      break;
    case 49:
      LOBYTE(v2) = sub_21BE230(a1);
      break;
    case 50:
      LODWORD(v2) = sub_21BE190(a1) ^ 1;
      break;
  }
  return v2;
}
