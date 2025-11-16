// Function: sub_14A8D80
// Address: 0x14a8d80
//
__int64 __fastcall sub_14A8D80(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  int v4; // eax
  __int64 v6; // r13
  unsigned int v8; // r8d
  __int64 v10; // r14
  int v11; // ecx
  unsigned __int8 v12; // al

  v4 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v4 <= 0x17u || (unsigned int)(v4 - 60) > 0xC )
    return 0;
  v8 = v4 - 24;
  *a4 = v4 - 24;
  v10 = **(_QWORD **)(a2 - 24);
  v11 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned __int8)v11 <= 0x17u )
  {
    if ( (unsigned __int8)v11 <= 0x10u )
    {
      switch ( v4 )
      {
        case '<':
          v6 = *(_QWORD *)(a1 - 24);
          if ( !v6 )
            BUG();
          if ( *(_BYTE *)(v6 + 16) <= 0x10u && v10 == *(_QWORD *)v6 )
            goto LABEL_15;
          v12 = sub_15FF7F0(*(_WORD *)(a1 + 18) & 0x7FFF);
          v6 = sub_15A4750(a3, v10, v12);
          goto LABEL_13;
        case '=':
          if ( !(unsigned __int8)sub_15FF7E0(*(_WORD *)(a1 + 18) & 0x7FFF) )
            return 0;
          v6 = sub_15A43B0(a3, v10, 0);
          goto LABEL_13;
        case '>':
          if ( !(unsigned __int8)sub_15FF7F0(*(_WORD *)(a1 + 18) & 0x7FFF) )
            return 0;
          v6 = sub_15A43B0(a3, v10, 1);
          goto LABEL_13;
        case '?':
          v6 = sub_15A3EC0(a3, **(_QWORD **)(a2 - 24), 1);
          goto LABEL_13;
        case '@':
          v6 = sub_15A3F70(a3, **(_QWORD **)(a2 - 24), 1);
          goto LABEL_13;
        case 'A':
          v6 = sub_15A4020(a3, **(_QWORD **)(a2 - 24), 1);
          goto LABEL_13;
        case 'B':
          v6 = sub_15A40D0(a3, **(_QWORD **)(a2 - 24), 1);
          goto LABEL_13;
        case 'C':
          v6 = sub_15A3E10(a3, **(_QWORD **)(a2 - 24), 1);
          goto LABEL_13;
        case 'D':
          v6 = sub_15A3D60(a3, **(_QWORD **)(a2 - 24), 1);
LABEL_13:
          if ( !v6 )
            return 0;
          v8 = *a4;
LABEL_15:
          if ( a3 != sub_15A46C0(v8, v6, *(_QWORD *)a3, 1) )
            return 0;
          return v6;
        default:
          return 0;
      }
    }
    return 0;
  }
  if ( (unsigned int)(v11 - 60) > 0xC )
    return 0;
  if ( v4 != v11 )
    return 0;
  v6 = *(_QWORD *)(a3 - 24);
  if ( v10 != *(_QWORD *)v6 )
    return 0;
  return v6;
}
