// Function: sub_214D1D0
// Address: 0x214d1d0
//
__int64 __fastcall sub_214D1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 result; // rax
  _DWORD *v9; // rdi
  __int64 (*v10)(void); // rdx
  _DWORD *v11; // r10
  unsigned int v12; // ebx
  unsigned __int8 v13; // al
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // r15d
  const __m128i *v25; // r13
  __int8 v26; // al
  char v27; // si
  unsigned int v28; // ebx
  char i; // r8
  unsigned int v30; // r14d
  char v31; // r8
  bool v32; // al
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rax
  void *v36; // rdx
  unsigned int v37; // r13d
  __int64 v38; // rax
  _WORD *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // r13
  unsigned int v42; // eax
  char v43; // di
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 v46; // [rsp+8h] [rbp-178h]
  __int64 v47; // [rsp+10h] [rbp-170h]
  unsigned int v48; // [rsp+20h] [rbp-160h]
  unsigned int v49; // [rsp+24h] [rbp-15Ch]
  __int64 v50; // [rsp+28h] [rbp-158h]
  int v51; // [rsp+28h] [rbp-158h]
  __m128i v52; // [rsp+30h] [rbp-150h] BYREF
  _BYTE *v53; // [rsp+40h] [rbp-140h] BYREF
  __int64 v54; // [rsp+48h] [rbp-138h]
  _BYTE v55[304]; // [rsp+50h] [rbp-130h] BYREF

  v7 = sub_396DDB0();
  result = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)result != 13 || (*(_BYTE *)(a2 + 9) & 1) != 0 )
  {
    v9 = *(_DWORD **)(a1 + 840);
    v10 = *(__int64 (**)(void))(*(_QWORD *)v9 + 56LL);
    if ( (char *)v10 == (char *)sub_214ABA0 )
    {
      v11 = v9 + 174;
    }
    else
    {
      v18 = v10();
      v9 = *(_DWORD **)(a1 + 840);
      v11 = (_DWORD *)v18;
      result = *(unsigned __int8 *)(a2 + 8);
    }
    v50 = (__int64)v11;
    if ( (_BYTE)result )
    {
      v12 = v9[63];
      sub_1263B40(a4, " (");
      if ( v12 > 0x13 )
      {
        v13 = *(_BYTE *)(a2 + 8);
        if ( (unsigned __int8)(v13 - 1) <= 5u )
          goto LABEL_7;
        if ( v13 == 11 )
        {
          if ( !sub_1642F90(a2, 128) )
          {
            if ( *(_BYTE *)(a2 + 8) == 11 )
            {
              v14 = *(_DWORD *)(a2 + 8) >> 8;
              goto LABEL_8;
            }
LABEL_7:
            v14 = (unsigned int)sub_1643030(a2);
LABEL_8:
            v15 = sub_1263B40(a4, ".param .b");
            v16 = 32;
            if ( (unsigned int)v14 >= 0x20 )
              v16 = v14;
            v17 = sub_16E7A90(v15, v16);
            sub_1263B40(v17, " func_retval0");
            return sub_1263B40(a4, ") ");
          }
          v13 = *(_BYTE *)(a2 + 8);
        }
        if ( v13 == 15 )
        {
          v41 = sub_1263B40(a4, ".param .b");
          v42 = 8 * sub_15A9520(v7, 0);
          if ( v42 == 32 )
          {
            v43 = 5;
          }
          else if ( v42 > 0x20 )
          {
            v43 = 6;
            if ( v42 != 64 )
            {
              v43 = 0;
              if ( v42 == 128 )
                v43 = 7;
            }
          }
          else
          {
            v43 = 3;
            if ( v42 != 8 )
              v43 = 4 * (v42 == 16);
          }
          v44 = sub_214AF00(v43);
          v45 = sub_16E7A90(v41, v44);
          sub_1263B40(v45, " func_retval0");
        }
        else
        {
          if ( (unsigned int)v13 - 13 > 1 && v13 != 16 )
            sub_1642F90(a2, 128);
          LODWORD(v53) = 0;
          v19 = sub_12BE0A0(v7, a2);
          if ( !(unsigned __int8)sub_1C2FF50(a3, 0, &v53) )
            LODWORD(v53) = sub_15A9FE0(v7, a2);
          v20 = sub_1263B40(a4, ".param .align ");
          v21 = sub_16E7A90(v20, (unsigned int)v53);
          v22 = sub_1263B40(v21, " .b8 func_retval0[");
          v23 = sub_16E7A90(v22, v19);
          sub_1263B40(v23, "]");
        }
        return sub_1263B40(a4, ") ");
      }
      v53 = v55;
      v54 = 0x1000000000LL;
      sub_20C7CE0(v50, v7, a2, (__int64)&v53, 0, 0);
      if ( !(_DWORD)v54 )
      {
LABEL_48:
        if ( v53 != v55 )
          _libc_free((unsigned __int64)v53);
        return sub_1263B40(a4, ") ");
      }
      v46 = (unsigned int)v54;
      v24 = 0;
      v48 = v54 - 1;
      v47 = 0;
      while ( 1 )
      {
        v25 = (const __m128i *)&v53[16 * v47];
        v52 = _mm_loadu_si128(v25);
        v26 = v25->m128i_i8[0];
        if ( v25->m128i_i8[0] )
        {
          if ( (unsigned __int8)(v26 - 14) > 0x5Fu )
            goto LABEL_27;
          v51 = word_4327020[(unsigned __int8)(v26 - 14)];
          switch ( v26 )
          {
            case 24:
            case 25:
            case 26:
            case 27:
            case 28:
            case 29:
            case 30:
            case 31:
            case 32:
            case 62:
            case 63:
            case 64:
            case 65:
            case 66:
            case 67:
              v27 = 3;
              v40 = 0;
              break;
            case 33:
            case 34:
            case 35:
            case 36:
            case 37:
            case 38:
            case 39:
            case 40:
            case 68:
            case 69:
            case 70:
            case 71:
            case 72:
            case 73:
              v27 = 4;
              v40 = 0;
              break;
            case 41:
            case 42:
            case 43:
            case 44:
            case 45:
            case 46:
            case 47:
            case 48:
            case 74:
            case 75:
            case 76:
            case 77:
            case 78:
            case 79:
              v27 = 5;
              v40 = 0;
              break;
            case 49:
            case 50:
            case 51:
            case 52:
            case 53:
            case 54:
            case 80:
            case 81:
            case 82:
            case 83:
            case 84:
            case 85:
              v27 = 6;
              v40 = 0;
              break;
            case 55:
              v27 = 7;
              v40 = 0;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v27 = 8;
              v40 = 0;
              break;
            case 89:
            case 90:
            case 91:
            case 92:
            case 93:
            case 101:
            case 102:
            case 103:
            case 104:
            case 105:
              v27 = 9;
              v40 = 0;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v27 = 10;
              v40 = 0;
              break;
            default:
              v27 = 2;
              v40 = 0;
              break;
          }
        }
        else
        {
          if ( !sub_1F58D20((__int64)&v53[16 * v47]) )
          {
LABEL_27:
            v49 = 0;
            v27 = v52.m128i_i8[0];
            v51 = 1;
            goto LABEL_28;
          }
          v51 = sub_1F58D30((__int64)v25);
          v27 = sub_1F596B0((__int64)v25);
        }
        v52.m128i_i8[0] = v27;
        v52.m128i_i64[1] = v40;
        if ( v51 )
          break;
LABEL_45:
        if ( v48 > (unsigned int)v47 )
          sub_1263B40(a4, ", ");
        if ( ++v47 == v46 )
          goto LABEL_48;
      }
      v49 = v51 - 1;
LABEL_28:
      v28 = 0;
      for ( i = v27; ; i = v52.m128i_i8[0] )
      {
        v37 = v24 + v28;
        if ( i )
        {
          v30 = sub_214AF00(i);
          v32 = (unsigned __int8)(v31 - 14) <= 0x47u || (unsigned __int8)(v31 - 2) <= 5u;
        }
        else
        {
          v30 = sub_1F58D40((__int64)&v52);
          v32 = sub_1F58CF0((__int64)&v52);
        }
        if ( v32 && v30 < 0x20 )
          v30 = 32;
        v33 = *(_QWORD *)(a4 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v33) <= 6 )
        {
          v34 = sub_16E7EE0(a4, ".reg .b", 7u);
        }
        else
        {
          *(_DWORD *)v33 = 1734701614;
          v34 = a4;
          *(_WORD *)(v33 + 4) = 11808;
          *(_BYTE *)(v33 + 6) = 98;
          *(_QWORD *)(a4 + 24) += 7LL;
        }
        v35 = sub_16E7A90(v34, v30);
        v36 = *(void **)(v35 + 24);
        if ( *(_QWORD *)(v35 + 16) - (_QWORD)v36 <= 0xBu )
        {
          v38 = sub_16E7EE0(v35, " func_retval", 0xCu);
          sub_16E7A90(v38, v37);
          if ( v49 <= v28 )
            goto LABEL_37;
        }
        else
        {
          qmemcpy(v36, " func_retval", 12);
          *(_QWORD *)(v35 + 24) += 12LL;
          sub_16E7A90(v35, v37);
          if ( v49 <= v28 )
            goto LABEL_37;
        }
        v39 = *(_WORD **)(a4 + 24);
        if ( *(_QWORD *)(a4 + 16) - (_QWORD)v39 <= 1u )
        {
          sub_16E7EE0(a4, ", ", 2u);
LABEL_37:
          if ( ++v28 == v51 )
            goto LABEL_44;
          continue;
        }
        ++v28;
        *v39 = 8236;
        *(_QWORD *)(a4 + 24) += 2LL;
        if ( v28 == v51 )
        {
LABEL_44:
          v24 += v28;
          goto LABEL_45;
        }
      }
    }
  }
  return result;
}
