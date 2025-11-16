// Function: sub_16ECBD0
// Address: 0x16ecbd0
//
__int64 __fastcall sub_16ECBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rbx
  unsigned __int64 v12; // r11
  _BYTE *v13; // r11
  __int64 v14; // rdx
  char v15; // bl
  char v16; // r12
  __int64 v17; // rbx
  _BYTE *v18; // rdi
  __int64 v19; // rbx
  char v20; // r11
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // r11
  __int64 v24; // [rsp+0h] [rbp-50h]
  char v25; // [rsp+13h] [rbp-3Dh]

  result = a6;
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(a1 + 8);
        v10 = a2;
        v11 = *(_QWORD *)(v9 + 8 * a2);
        v12 = (unsigned int)v11 & 0xF8000000;
        if ( v12 != 1476395008 )
          break;
LABEL_46:
        *(_BYTE *)(a6 + a2 + 1) |= *(_BYTE *)(a6 + a2);
        *(_BYTE *)(a6 + (v11 & 0x7FFFFFF) + a2) |= *(_BYTE *)(a6 + a2);
        if ( a3 == ++a2 )
          return result;
      }
      if ( v12 > 0x58000000 )
      {
        if ( v12 == 2281701376 )
        {
          v18 = (_BYTE *)(a6 + a2);
          v19 = a2 + (v11 & 0x7FFFFFF);
          v20 = *(_BYTE *)(a6 + a2++);
          *(_BYTE *)(a6 + a2) |= v20;
          if ( (*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v19) & 0xF8000000LL) == 0x90000000LL )
            goto LABEL_11;
          *(_BYTE *)(a6 + v19) |= *v18;
          if ( a3 == a2 )
            return result;
        }
        else if ( v12 > 0x88000000 )
        {
          switch ( v12 )
          {
            case 0x98000000uLL:
              ++a2;
              if ( a5 == 133 )
              {
LABEL_10:
                *(_BYTE *)(a6 + v10 + 1) |= *(_BYTE *)(a4 + v10);
                goto LABEL_11;
              }
              if ( a3 == a2 )
                return result;
              break;
            case 0xA0000000uLL:
              ++a2;
              if ( a5 == 134 )
                goto LABEL_10;
              if ( a3 == a2 )
                return result;
              break;
            case 0x90000000uLL:
              goto LABEL_43;
            default:
              goto LABEL_26;
          }
        }
        else
        {
          if ( v12 == 2013265920 )
            goto LABEL_46;
          if ( v12 <= 0x78000000 )
          {
            if ( ((v12 - 1744830464) & 0xFFFFFFFFF0000000LL) != 0 && v12 != 1610612736 )
              goto LABEL_26;
LABEL_43:
            *(_BYTE *)(a6 + a2 + 1) |= *(_BYTE *)(a6 + a2);
            ++a2;
            goto LABEL_23;
          }
          if ( v12 != 0x80000000 )
            goto LABEL_26;
          v24 = a2 + 1;
          v25 = *(_BYTE *)(a6 + a2);
          if ( v25 )
          {
            v21 = *(_QWORD *)(v9 + 8 * a2 + 8);
            v22 = a2 + 1;
            if ( (v21 & 0xF8000000) != 0x90000000LL )
            {
              v23 = 1;
              do
              {
                v23 += v21 & 0x7FFFFFF;
                v22 = v23 + a2;
                v21 = *(_QWORD *)(v9 + 8 * (v23 + a2));
              }
              while ( (v21 & 0xF8000000) != 0x90000000LL );
            }
            ++a2;
            *(_BYTE *)(a6 + v22) |= v25;
            if ( a3 == v24 )
              return result;
          }
          else
          {
            ++a2;
            if ( a3 == v24 )
              return result;
          }
        }
      }
      else
      {
        if ( v12 == 805306368 )
        {
          ++a2;
          if ( a5 <= 127 )
          {
            v17 = *(_QWORD *)(a1 + 24) + 32 * (v11 & 0x7FFFFFF);
            if ( (*(_BYTE *)(v17 + 8) & *(_BYTE *)(*(_QWORD *)v17 + (unsigned __int8)a5)) != 0 )
              *(_BYTE *)(a6 + v10 + 1) |= *(_BYTE *)(a4 + v10);
          }
          goto LABEL_11;
        }
        if ( v12 > 0x30000000 )
        {
          if ( v12 == 1207959552 )
            goto LABEL_43;
          if ( v12 <= 0x48000000 )
          {
            if ( ((v12 - 939524096) & 0xFFFFFFFFF0000000LL) != 0 )
              goto LABEL_26;
            goto LABEL_43;
          }
          if ( v12 != 1342177280 )
            goto LABEL_26;
          v13 = (_BYTE *)(a6 + a2);
          v14 = a2 + 1;
          a2 -= v11 & 0x7FFFFFF;
          *(_BYTE *)(a6 + v14) |= *v13;
          v15 = *(_BYTE *)(a6 + a2);
          v16 = *v13 | v15;
          *(_BYTE *)(a6 + a2) = v16;
          if ( v16 && !v15 )
            goto LABEL_11;
          a2 = v14;
LABEL_23:
          if ( a3 == a2 )
            return result;
        }
        else if ( v12 == 0x20000000 )
        {
          ++a2;
          if ( (unsigned int)(a5 - 130) <= 1 )
            goto LABEL_10;
          if ( a3 == a2 )
            return result;
        }
        else if ( v12 <= 0x20000000 )
        {
          if ( v12 == 0x10000000 )
          {
            ++a2;
            if ( (char)v11 == a5 )
              goto LABEL_10;
            if ( a3 == a2 )
              return result;
          }
          else
          {
            if ( v12 == 402653184 )
            {
              ++a2;
              if ( (a5 & 0xFFFFFFFD) == 0x81 )
                *(_BYTE *)(a6 + v10 + 1) |= *(_BYTE *)(a4 + v10);
              goto LABEL_11;
            }
LABEL_26:
            if ( a3 == ++a2 )
              return result;
          }
        }
        else
        {
          if ( v12 != 671088640 )
            goto LABEL_26;
          ++a2;
          if ( a5 <= 127 )
            goto LABEL_10;
LABEL_11:
          if ( a3 == a2 )
            return result;
        }
      }
    }
  }
  return result;
}
