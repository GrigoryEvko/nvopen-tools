// Function: sub_3734910
// Address: 0x3734910
//
unsigned __int64 __fastcall sub_3734910(int *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  _QWORD *v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int16 v6; // ax
  int *v7; // rax
  size_t v8; // rdx
  __int64 v9; // rbx
  int v11[8]; // [rsp+Fh] [rbp-21h] BYREF

  sub_372FCB0(a1, 0x44u);
  sub_372FCB0(a1, *(unsigned __int16 *)(a2 + 28));
  sub_3734770((__int64)a1, a2, v2, v3);
  v4 = *(_QWORD **)(a2 + 32);
  if ( v4 )
  {
    v5 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5 )
    {
      while ( 1 )
      {
        v6 = *(_WORD *)(v5 + 28);
        if ( v6 <= 0x4Bu )
          break;
LABEL_6:
        if ( v6 != 17152 && (unsigned __int16)(v6 + 20479) > 3u )
          goto LABEL_9;
LABEL_8:
        v7 = (int *)sub_372FC20(v5);
        if ( !v8 )
          goto LABEL_9;
        sub_37333A0(a1, v5, v7, v8);
LABEL_10:
        v9 = *(_QWORD *)v5;
        if ( (v9 & 4) == 0 )
        {
          v5 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v5 )
            continue;
        }
        goto LABEL_12;
      }
      if ( v6 )
      {
        switch ( v6 )
        {
          case 1u:
          case 2u:
          case 4u:
          case 0xFu:
          case 0x10u:
          case 0x12u:
          case 0x13u:
          case 0x15u:
          case 0x16u:
          case 0x17u:
          case 0x1Fu:
          case 0x20u:
          case 0x21u:
          case 0x24u:
          case 0x26u:
          case 0x29u:
          case 0x2Du:
          case 0x31u:
          case 0x35u:
          case 0x37u:
          case 0x38u:
          case 0x3Bu:
          case 0x40u:
          case 0x42u:
          case 0x43u:
          case 0x44u:
          case 0x46u:
          case 0x47u:
          case 0x4Bu:
            goto LABEL_8;
          default:
            if ( v6 != 46 )
              break;
            v6 = *(_WORD *)(sub_3214EE0(v5) + 28);
            if ( v6 > 0x4Bu )
              goto LABEL_6;
            if ( v6 )
            {
              switch ( v6 )
              {
                case 1u:
                case 2u:
                case 4u:
                case 0xFu:
                case 0x10u:
                case 0x12u:
                case 0x13u:
                case 0x15u:
                case 0x16u:
                case 0x17u:
                case 0x1Fu:
                case 0x20u:
                case 0x21u:
                case 0x24u:
                case 0x26u:
                case 0x29u:
                case 0x2Du:
                case 0x31u:
                case 0x35u:
                case 0x37u:
                case 0x38u:
                case 0x3Bu:
                case 0x40u:
                case 0x42u:
                case 0x43u:
                case 0x44u:
                case 0x46u:
                case 0x47u:
                case 0x4Bu:
                  goto LABEL_8;
                default:
                  goto LABEL_9;
              }
            }
            break;
        }
      }
LABEL_9:
      sub_3734910(a1, v5);
      goto LABEL_10;
    }
  }
LABEL_12:
  LOBYTE(v11[0]) = 0;
  return sub_C7D060(a1, v11, 1u);
}
