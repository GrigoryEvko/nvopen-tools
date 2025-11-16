// Function: sub_284F800
// Address: 0x284f800
//
__int64 __fastcall sub_284F800(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v8; // rax
  unsigned int v9; // eax

  v4 = (_QWORD *)sub_BD5C60(a2);
  v5 = sub_BCB120(v4);
  v6 = sub_B46690(a2);
  if ( !v6 )
    v6 = v5;
  if ( *(_BYTE *)a2 == 85 )
  {
    v8 = *(_QWORD *)(a2 - 32);
    if ( v8 )
    {
      if ( !*(_BYTE *)v8 && *(_QWORD *)(v8 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
      {
        v9 = *(_DWORD *)(v8 + 36);
        if ( v9 > 0xF3 )
        {
          if ( v9 == 286 )
            return *(_QWORD *)(a3 + 8);
        }
        else if ( v9 > 0xE3 )
        {
          switch ( v9 )
          {
            case 0xE4u:
            case 0xE6u:
              return v6;
            case 0xEEu:
            case 0xF1u:
              v6 = *(_QWORD *)(a3 + 8);
              break;
            case 0xF3u:
              return *(_QWORD *)(a3 + 8);
            default:
              goto LABEL_13;
          }
          return v6;
        }
LABEL_13:
        sub_DFDD50(a1);
      }
    }
  }
  return v6;
}
