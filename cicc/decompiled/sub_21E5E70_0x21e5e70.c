// Function: sub_21E5E70
// Address: 0x21e5e70
//
unsigned __int64 __fastcall sub_21E5E70(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 result; // rax
  _DWORD *v6; // rdx
  __int64 v7; // rdx
  _DWORD *v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  size_t v21; // rdx
  char *v22; // rsi

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  result = (unsigned __int8)v4 >> 4;
  if ( (((unsigned int)v4 >> 4) & 0xF) == 1 )
  {
    v8 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v8;
    if ( result <= 3 )
    {
      result = sub_16E7EE0(a3, ".cta", 4u);
    }
    else
    {
      *v8 = 1635017518;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  else if ( (_BYTE)result == 2 )
  {
    v6 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v6;
    if ( result <= 3 )
    {
      result = sub_16E7EE0(a3, ".sys", 4u);
    }
    else
    {
      *v6 = 1937339182;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  switch ( BYTE2(v4) )
  {
    case 0:
      v9 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v9) <= 6 )
      {
        v21 = 7;
        v22 = ".exch.b";
        goto LABEL_37;
      }
      *(_DWORD *)v9 = 1668834606;
      *(_WORD *)(v9 + 4) = 11880;
      *(_BYTE *)(v9 + 6) = 98;
      *(_QWORD *)(a3 + 24) += 7LL;
      result = 11880;
      break;
    case 1:
      v10 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v10) <= 5 )
      {
        v21 = 6;
        v22 = ".add.u";
        goto LABEL_37;
      }
      *(_DWORD *)v10 = 1684300078;
      *(_WORD *)(v10 + 4) = 29998;
      *(_QWORD *)(a3 + 24) += 6LL;
      result = 29998;
      break;
    case 3:
      v11 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v11) <= 5 )
      {
        v21 = 6;
        v22 = ".and.b";
        goto LABEL_37;
      }
      *(_DWORD *)v11 = 1684955438;
      *(_WORD *)(v11 + 4) = 25134;
      *(_QWORD *)(a3 + 24) += 6LL;
      result = 25134;
      break;
    case 5:
      v12 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v12;
      if ( result <= 4 )
      {
        v21 = 5;
        v22 = ".or.b";
        goto LABEL_37;
      }
      *(_DWORD *)v12 = 779251502;
      *(_BYTE *)(v12 + 4) = 98;
      *(_QWORD *)(a3 + 24) += 5LL;
      break;
    case 6:
      v13 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v13;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".xor.b";
        goto LABEL_37;
      }
      *(_DWORD *)v13 = 1919907886;
      *(_WORD *)(v13 + 4) = 25134;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 7:
      v14 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v14;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".max.s";
        goto LABEL_37;
      }
      *(_DWORD *)v14 = 2019650862;
      *(_WORD *)(v14 + 4) = 29486;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 8:
      v17 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v17;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".min.s";
        goto LABEL_37;
      }
      *(_DWORD *)v17 = 1852402990;
      *(_WORD *)(v17 + 4) = 29486;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 9:
      v18 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v18;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".max.u";
        goto LABEL_37;
      }
      *(_DWORD *)v18 = 2019650862;
      *(_WORD *)(v18 + 4) = 29998;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 0xA:
      v19 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v19;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".min.u";
        goto LABEL_37;
      }
      *(_DWORD *)v19 = 1852402990;
      *(_WORD *)(v19 + 4) = 29998;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 0xB:
      v20 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v20;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".add.f";
        goto LABEL_37;
      }
      *(_DWORD *)v20 = 1684300078;
      *(_WORD *)(v20 + 4) = 26158;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 0xC:
      v15 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v15;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".inc.u";
        goto LABEL_37;
      }
      *(_DWORD *)v15 = 1668180270;
      *(_WORD *)(v15 + 4) = 29998;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 0xD:
      v16 = *(_QWORD *)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - v16;
      if ( result <= 5 )
      {
        v21 = 6;
        v22 = ".dec.u";
        goto LABEL_37;
      }
      *(_DWORD *)v16 = 1667589166;
      *(_WORD *)(v16 + 4) = 29998;
      *(_QWORD *)(a3 + 24) += 6LL;
      break;
    case 0xE:
      v7 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v7) <= 5 )
      {
        v21 = 6;
        v22 = ".cas.b";
LABEL_37:
        result = sub_16E7EE0(a3, v22, v21);
      }
      else
      {
        *(_DWORD *)v7 = 1935762222;
        *(_WORD *)(v7 + 4) = 25134;
        *(_QWORD *)(a3 + 24) += 6LL;
        result = 25134;
      }
      break;
    default:
      return result;
  }
  return result;
}
