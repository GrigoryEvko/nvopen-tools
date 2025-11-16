// Function: sub_35F02F0
// Address: 0x35f02f0
//
unsigned __int64 __fastcall sub_35F02F0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rbx
  char v6; // al
  unsigned __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  _DWORD *v10; // rdx
  _DWORD *v11; // rdx
  _QWORD *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  size_t v25; // rdx
  char *v26; // rsi

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( (v5 & 0x200) != 0 )
  {
    v9 = *(_QWORD *)(a4 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v9) <= 8 )
    {
      sub_CB6200(a4, (unsigned __int8 *)"::cluster", 9u);
    }
    else
    {
      *(_BYTE *)(v9 + 8) = 114;
      *(_QWORD *)v9 = 0x657473756C633A3ALL;
      *(_QWORD *)(a4 + 32) += 9LL;
    }
  }
  v6 = (unsigned __int8)v5 >> 4;
  if ( (unsigned __int8)v5 >> 4 == 2 )
  {
    v10 = *(_DWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v10 <= 3u )
    {
      sub_CB6200(a4, (unsigned __int8 *)".sys", 4u);
    }
    else
    {
      *v10 = 1937339182;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  else if ( v6 == 3 )
  {
    v12 = *(_QWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v12 <= 7u )
    {
      sub_CB6200(a4, (unsigned __int8 *)".cluster", 8u);
    }
    else
    {
      *v12 = 0x72657473756C632ELL;
      *(_QWORD *)(a4 + 32) += 8LL;
    }
  }
  else if ( v6 == 1 )
  {
    v11 = *(_DWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v11 <= 3u )
    {
      sub_CB6200(a4, (unsigned __int8 *)".cta", 4u);
    }
    else
    {
      *v11 = 1635017518;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  result = sub_35ED4A0(v5 & 0xF, a4);
  switch ( BYTE2(v5) )
  {
    case 0:
      v13 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v13) <= 6 )
      {
        v25 = 7;
        v26 = ".exch.b";
        goto LABEL_45;
      }
      *(_DWORD *)v13 = 1668834606;
      *(_WORD *)(v13 + 4) = 11880;
      *(_BYTE *)(v13 + 6) = 98;
      *(_QWORD *)(a4 + 32) += 7LL;
      result = 11880;
      break;
    case 1:
      v14 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v14) <= 5 )
      {
        v25 = 6;
        v26 = ".add.u";
        goto LABEL_45;
      }
      *(_DWORD *)v14 = 1684300078;
      *(_WORD *)(v14 + 4) = 29998;
      *(_QWORD *)(a4 + 32) += 6LL;
      result = 29998;
      break;
    case 3:
      v15 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v15) <= 5 )
      {
        v25 = 6;
        v26 = ".and.b";
        goto LABEL_45;
      }
      *(_DWORD *)v15 = 1684955438;
      *(_WORD *)(v15 + 4) = 25134;
      *(_QWORD *)(a4 + 32) += 6LL;
      result = 25134;
      break;
    case 5:
      v16 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v16;
      if ( result <= 4 )
      {
        v25 = 5;
        v26 = ".or.b";
        goto LABEL_45;
      }
      *(_DWORD *)v16 = 779251502;
      *(_BYTE *)(v16 + 4) = 98;
      *(_QWORD *)(a4 + 32) += 5LL;
      break;
    case 6:
      v17 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v17;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".xor.b";
        goto LABEL_45;
      }
      *(_DWORD *)v17 = 1919907886;
      *(_WORD *)(v17 + 4) = 25134;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 7:
      v18 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v18;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".max.s";
        goto LABEL_45;
      }
      *(_DWORD *)v18 = 2019650862;
      *(_WORD *)(v18 + 4) = 29486;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 8:
      v19 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v19;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".min.s";
        goto LABEL_45;
      }
      *(_DWORD *)v19 = 1852402990;
      *(_WORD *)(v19 + 4) = 29486;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 9:
      v20 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v20;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".max.u";
        goto LABEL_45;
      }
      *(_DWORD *)v20 = 2019650862;
      *(_WORD *)(v20 + 4) = 29998;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 0xA:
      v21 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v21;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".min.u";
        goto LABEL_45;
      }
      *(_DWORD *)v21 = 1852402990;
      *(_WORD *)(v21 + 4) = 29998;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 0xB:
      v22 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v22;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".add.f";
        goto LABEL_45;
      }
      *(_DWORD *)v22 = 1684300078;
      *(_WORD *)(v22 + 4) = 26158;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 0xC:
      v23 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v23;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".inc.u";
        goto LABEL_45;
      }
      *(_DWORD *)v23 = 1668180270;
      *(_WORD *)(v23 + 4) = 29998;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 0xD:
      v24 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v24;
      if ( result <= 5 )
      {
        v25 = 6;
        v26 = ".dec.u";
        goto LABEL_45;
      }
      *(_DWORD *)v24 = 1667589166;
      *(_WORD *)(v24 + 4) = 29998;
      *(_QWORD *)(a4 + 32) += 6LL;
      break;
    case 0xE:
      v8 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v8) <= 5 )
      {
        v25 = 6;
        v26 = ".cas.b";
LABEL_45:
        result = sub_CB6200(a4, (unsigned __int8 *)v26, v25);
      }
      else
      {
        *(_DWORD *)v8 = 1935762222;
        *(_WORD *)(v8 + 4) = 25134;
        *(_QWORD *)(a4 + 32) += 6LL;
        result = 25134;
      }
      break;
    default:
      return result;
  }
  return result;
}
