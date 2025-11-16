// Function: sub_CBAFE0
// Address: 0xcbafe0
//
unsigned __int64 __fastcall sub_CBAFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned __int64 a6)
{
  unsigned __int64 v7; // rdx
  __int64 v8; // r11
  __int64 v9; // rcx
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  char v18; // cl
  __int64 v19; // rcx
  unsigned __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rcx

  v7 = 1LL << a2;
  if ( a2 != a3 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    while ( 1 )
    {
      v9 = *(_QWORD *)(v8 + 8 * a2);
      v10 = a2;
      v11 = v7;
      v12 = (unsigned int)v9 & 0xF8000000;
      if ( v12 == 1476395008 )
        goto LABEL_46;
      if ( v12 > 0x58000000 )
      {
        if ( v12 == 0x80000000 )
        {
          v21 = a2 + 1;
          if ( (a6 & v7) != 0 )
          {
            v23 = *(_QWORD *)(v8 + 8 * a2 + 8);
            if ( (v23 & 0xF8000000) == 0x90000000LL )
            {
              LOBYTE(v24) = 1;
            }
            else
            {
              v24 = 1;
              do
              {
                v24 += v23 & 0x7FFFFFF;
                v23 = *(_QWORD *)(v8 + 8 * (v24 + a2));
              }
              while ( (v23 & 0xF8000000) != 0x90000000LL );
            }
            ++a2;
            a6 |= (a6 & v7) << v24;
            v7 *= 2LL;
            if ( a3 == v21 )
              return a6;
          }
          else
          {
            ++a2;
            v7 *= 2LL;
            if ( a3 == v21 )
              return a6;
          }
        }
        else if ( v12 <= 0x80000000 )
        {
          if ( v12 == 1879048192 )
            goto LABEL_26;
          if ( v12 <= 0x70000000 )
          {
            if ( (v9 & 0xF0000000) == 0x60000000 )
              goto LABEL_26;
            goto LABEL_22;
          }
          if ( v12 != 2013265920 )
            goto LABEL_22;
LABEL_46:
          ++a2;
          a6 |= ((((2 * (v7 & a6)) | a6) & v7) << v9) | (2 * (v7 & a6));
          v7 *= 2LL;
          if ( a3 == a2 )
            return a6;
        }
        else if ( v12 == 2550136832 )
        {
          ++a2;
          if ( a5 == 133 )
            goto LABEL_11;
          v7 *= 2LL;
          if ( a3 == a2 )
            return a6;
        }
        else if ( v12 <= 0x98000000 )
        {
          if ( v12 == 2281701376 )
          {
            v19 = v9 & 0x7FFFFFF;
            ++a2;
            a6 |= 2 * (v7 & a6);
            if ( (*(_QWORD *)(v8 + 8 * (v19 + v10)) & 0xF8000000LL) == 0x90000000LL )
              goto LABEL_12;
            v20 = v7 & a6;
            v7 *= 2LL;
            a6 |= v20 << v19;
            if ( a3 == a2 )
              return a6;
          }
          else
          {
            if ( v12 == 2415919104 )
            {
LABEL_26:
              ++a2;
              a6 |= 2 * (a6 & v7);
              goto LABEL_23;
            }
LABEL_22:
            ++a2;
LABEL_23:
            v7 = 2 * v11;
            if ( a3 == a2 )
              return a6;
          }
        }
        else
        {
          if ( v12 != 2684354560 )
            goto LABEL_22;
          ++a2;
          if ( a5 == 134 )
            goto LABEL_11;
          v7 *= 2LL;
          if ( a3 == a2 )
            return a6;
        }
      }
      else
      {
        if ( v12 == 805306368 )
        {
          ++a2;
          if ( a5 <= 127 )
          {
            v22 = *(_QWORD *)(a1 + 24) + 32 * (v9 & 0x7FFFFFF);
            if ( (*(_BYTE *)(v22 + 8) & *(_BYTE *)(*(_QWORD *)v22 + (unsigned __int8)a5)) != 0 )
              a6 |= 2 * (v7 & a4);
          }
          goto LABEL_12;
        }
        if ( v12 > 0x30000000 )
        {
          if ( v12 == 1207959552 )
            goto LABEL_26;
          if ( v12 <= 0x48000000 )
          {
            if ( ((v12 - 939524096) & 0xFFFFFFFFF0000000LL) == 0 )
              goto LABEL_26;
            goto LABEL_22;
          }
          if ( v12 != 1342177280 )
            goto LABEL_22;
          v14 = v7 & a6;
          v15 = v7 >> v9;
          v16 = a6 | (2 * v14);
          a6 = v16 | ((v11 & v16) >> v9);
          if ( (v15 & v16) != 0 || (a6 & v15) == 0 )
            goto LABEL_22;
          v17 = a2 - 1 - (v9 & 0x7FFFFFF);
          v18 = v17;
          a2 = v17 + 1;
          v7 = 2 * (1LL << v18);
          if ( a3 == a2 )
            return a6;
        }
        else if ( v12 == 0x20000000 )
        {
          ++a2;
          if ( (unsigned int)(a5 - 130) <= 1 )
            goto LABEL_11;
          v7 *= 2LL;
          if ( a3 == a2 )
            return a6;
        }
        else if ( v12 <= 0x20000000 )
        {
          if ( v12 != 0x10000000 )
          {
            if ( v12 == 402653184 )
            {
              ++a2;
              if ( (a5 & 0xFFFFFFFD) == 0x81 )
                a6 |= 2 * (v7 & a4);
              goto LABEL_12;
            }
            goto LABEL_22;
          }
          ++a2;
          if ( (char)v9 == a5 )
          {
LABEL_11:
            a6 |= 2 * (v7 & a4);
            goto LABEL_12;
          }
          v7 *= 2LL;
          if ( a3 == a2 )
            return a6;
        }
        else
        {
          if ( v12 != 671088640 )
            goto LABEL_22;
          ++a2;
          if ( a5 <= 127 )
            goto LABEL_11;
LABEL_12:
          v7 *= 2LL;
          if ( a3 == a2 )
            return a6;
        }
      }
    }
  }
  return a6;
}
