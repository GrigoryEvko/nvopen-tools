// Function: sub_2C15520
// Address: 0x2c15520
//
unsigned __int64 __fastcall sub_2C15520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r11
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r9
  unsigned __int64 v14; // rdx
  __int64 v15; // r13
  _BYTE *v16; // rsi
  int v17; // edi
  _BYTE *v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  _BOOL4 v33; // [rsp+20h] [rbp-80h]
  unsigned __int64 v34; // [rsp+20h] [rbp-80h]
  __int64 v35; // [rsp+30h] [rbp-70h]
  _BYTE *v36; // [rsp+40h] [rbp-60h] BYREF
  __int64 v37; // [rsp+48h] [rbp-58h]
  _BYTE v38[80]; // [rsp+50h] [rbp-50h] BYREF

  switch ( *(_DWORD *)(a1 + 160) )
  {
    case 0xC:
      v26 = sub_2BFD6A0(a3 + 16, a1 + 96);
      v27 = sub_2AAEDF0(v26, a2);
      result = sub_DFD800(*(_QWORD *)a3, *(_DWORD *)(a1 + 160), v27, *(_DWORD *)(a3 + 176), 0, 0, 0, 0, 0, 0);
      break;
    case 0xD:
    case 0xE:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x15:
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1D:
    case 0x1E:
      v5 = 0;
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
      if ( sub_2BF04A0(v6) || (v28 = sub_DFB770(*(unsigned __int8 **)(v6 + 40)), v33 = v28, v5 = v28, !(_DWORD)v28) )
        v33 = sub_2BFB0D0(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL));
      v7 = sub_2BFD6A0(a3 + 16, a1 + 96);
      v8 = sub_2AAEDF0(v7, a2);
      v9 = *(_QWORD *)(a1 + 136);
      v10 = v8;
      if ( v9 )
      {
        if ( *(_BYTE *)v9 <= 0x1Cu )
        {
          v20 = 0;
          v9 = 0;
          v36 = v38;
          v18 = v38;
          v37 = 0x400000000LL;
        }
        else
        {
          v36 = v38;
          v37 = 0x400000000LL;
          if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
          {
            v11 = *(_QWORD *)(v9 - 8);
            v12 = v11 + 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
          }
          else
          {
            v12 = v9;
            v11 = v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
          }
          v13 = v12 - v11;
          v14 = v13 >> 5;
          v15 = v13 >> 5;
          if ( (unsigned __int64)v13 > 0x80 )
          {
            v35 = v13 >> 5;
            v29 = v11;
            v30 = v13;
            v31 = v9;
            v32 = v10;
            sub_C8D5F0((__int64)&v36, v38, v14, 8u, v9, v13);
            v18 = v36;
            v17 = v37;
            LODWORD(v14) = v35;
            v10 = v32;
            v9 = v31;
            v13 = v30;
            v16 = &v36[8 * (unsigned int)v37];
            v11 = v29;
          }
          else
          {
            v16 = v38;
            v17 = 0;
            v18 = v38;
          }
          if ( v13 > 0 )
          {
            v19 = 0;
            do
            {
              *(_QWORD *)&v16[v19] = *(_QWORD *)(v11 + 4 * v19);
              v19 += 8;
              --v15;
            }
            while ( v15 );
            v17 = v37;
            v18 = v36;
          }
          LODWORD(v37) = v17 + v14;
          v20 = (unsigned int)(v17 + v14);
        }
      }
      else
      {
        v20 = 0;
        v36 = v38;
        v18 = v38;
        v37 = 0x400000000LL;
      }
      result = sub_DFD800(
                 *(_QWORD *)a3,
                 *(_DWORD *)(a1 + 160),
                 v10,
                 *(_DWORD *)(a3 + 176),
                 0,
                 v5 & 0xFFFFFFFF00000000LL | v33,
                 v18,
                 v20,
                 v9,
                 *(__int64 **)(a3 + 8));
      if ( v36 != v38 )
      {
        v34 = result;
        _libc_free((unsigned __int64)v36);
        result = v34;
      }
      break;
    case 0x13:
    case 0x14:
    case 0x16:
    case 0x17:
      result = sub_2AD2480(a3, *(unsigned __int8 **)(a1 + 136), a2);
      break;
    case 0x35:
    case 0x36:
      v22 = sub_2BFD6A0(a3 + 16, **(_QWORD **)(a1 + 48));
      v23 = sub_2AAEDF0(v22, a2);
      result = sub_DFD2D0(*(__int64 **)a3, *(unsigned int *)(a1 + 160), v23);
      break;
    case 0x40:
      result = sub_DFD440(*(_QWORD *)a3, 64, *(_DWORD *)(a3 + 176));
      break;
    case 0x43:
      v24 = sub_2BFD6A0(a3 + 16, a1 + 96);
      v25 = sub_2AAEDF0(v24, a2);
      result = sub_DFD800(*(_QWORD *)a3, 0x11u, v25, *(_DWORD *)(a3 + 176), 0, 0, 0, 0, 0, 0);
      break;
    default:
      BUG();
  }
  return result;
}
