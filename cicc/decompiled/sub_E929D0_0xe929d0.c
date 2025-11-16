// Function: sub_E929D0
// Address: 0xe929d0
//
_BYTE *__fastcall sub_E929D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v7; // rdx
  __int64 v8; // r14
  _WORD *v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int8 *v12; // rsi
  unsigned __int64 v13; // rax
  int v14; // eax
  _BYTE *v15; // rdx
  unsigned __int64 v16; // rcx
  int v17; // eax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  unsigned __int64 v21; // rax
  _BYTE *v22; // rdi
  unsigned __int64 v23; // r13
  unsigned __int8 *v24; // rsi
  _BYTE *result; // rax
  __int64 v26; // rdx
  _BYTE *v27; // rax
  _BYTE *v28; // rax
  _BYTE *v29; // rdx
  __int64 v30; // rcx
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax

  if ( (unsigned __int8)sub_E92970(a1, *(_QWORD *)(a1 + 128), *(_QWORD *)(a1 + 136)) )
  {
    v20 = *(_BYTE **)(a4 + 32);
    if ( (unsigned __int64)v20 >= *(_QWORD *)(a4 + 24) )
    {
      a4 = sub_CB5D20(a4, 9);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v20 + 1;
      *v20 = 9;
    }
    v21 = *(_QWORD *)(a4 + 24);
    v22 = *(_BYTE **)(a4 + 32);
    v23 = *(_QWORD *)(a1 + 136);
    v24 = *(unsigned __int8 **)(a1 + 128);
    if ( v23 > v21 - (unsigned __int64)v22 )
    {
      v37 = sub_CB6200(a4, v24, *(_QWORD *)(a1 + 136));
      v22 = *(_BYTE **)(v37 + 32);
      a4 = v37;
      v21 = *(_QWORD *)(v37 + 24);
    }
    else if ( v23 )
    {
      memcpy(v22, v24, *(_QWORD *)(a1 + 136));
      v21 = *(_QWORD *)(a4 + 24);
      v22 = (_BYTE *)(v23 + *(_QWORD *)(a4 + 32));
      *(_QWORD *)(a4 + 32) = v22;
    }
    if ( v21 > (unsigned __int64)v22 )
    {
      *(_QWORD *)(a4 + 32) = v22 + 1;
      *v22 = 10;
      return v22 + 1;
    }
    return (_BYTE *)sub_CB5D20(a4, 10);
  }
  v7 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v7 <= 9u )
  {
    v10 = sub_CB6200(a4, "\t.section\t", 0xAu);
    v9 = *(_WORD **)(v10 + 32);
    v8 = v10;
  }
  else
  {
    v8 = a4;
    qmemcpy(v7, "\t.section\t", 10);
    v9 = (_WORD *)(*(_QWORD *)(a4 + 32) + 10LL);
    *(_QWORD *)(a4 + 32) = v9;
  }
  v11 = *(_QWORD *)(a1 + 136);
  v12 = *(unsigned __int8 **)(a1 + 128);
  v13 = *(_QWORD *)(v8 + 24) - (_QWORD)v9;
  if ( v11 > v13 )
  {
    v34 = sub_CB6200(v8, v12, *(_QWORD *)(a1 + 136));
    v9 = *(_WORD **)(v34 + 32);
    v8 = v34;
    v13 = *(_QWORD *)(v34 + 24) - (_QWORD)v9;
  }
  else if ( v11 )
  {
    memcpy(v9, v12, *(_QWORD *)(a1 + 136));
    v36 = *(_QWORD *)(v8 + 24);
    v9 = (_WORD *)(v11 + *(_QWORD *)(v8 + 32));
    *(_QWORD *)(v8 + 32) = v9;
    v13 = v36 - (_QWORD)v9;
  }
  if ( v13 <= 1 )
  {
    sub_CB6200(v8, (unsigned __int8 *)",\"", 2u);
    v14 = *(_DWORD *)(a1 + 148);
    if ( (v14 & 0x40) == 0 )
      goto LABEL_10;
  }
  else
  {
    *v9 = 8748;
    *(_QWORD *)(v8 + 32) += 2LL;
    v14 = *(_DWORD *)(a1 + 148);
    if ( (v14 & 0x40) == 0 )
      goto LABEL_10;
  }
  v32 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v32 < *(_QWORD *)(a4 + 24) )
  {
    *(_QWORD *)(a4 + 32) = v32 + 1;
    *v32 = 100;
    v14 = *(_DWORD *)(a1 + 148);
    if ( (v14 & 0x80u) == 0 )
      goto LABEL_11;
    goto LABEL_54;
  }
  sub_CB5D20(a4, 100);
  v14 = *(_DWORD *)(a1 + 148);
LABEL_10:
  if ( (v14 & 0x80u) == 0 )
    goto LABEL_11;
LABEL_54:
  v33 = *(_BYTE **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v33 )
  {
    sub_CB5D20(a4, 98);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v33 + 1;
    *v33 = 98;
  }
  v14 = *(_DWORD *)(a1 + 148);
LABEL_11:
  if ( (v14 & 0x20000000) == 0 )
    goto LABEL_12;
  v31 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v31 >= *(_QWORD *)(a4 + 24) )
  {
    sub_CB5D20(a4, 120);
    v14 = *(_DWORD *)(a1 + 148);
LABEL_12:
    v15 = *(_BYTE **)(a4 + 32);
    v16 = *(_QWORD *)(a4 + 24);
    if ( v14 >= 0 )
      goto LABEL_13;
LABEL_49:
    if ( v16 <= (unsigned __int64)v15 )
    {
      sub_CB5D20(a4, 119);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v15 + 1;
      *v15 = 119;
    }
LABEL_31:
    v17 = *(_DWORD *)(a1 + 148);
    if ( (v17 & 0x800) == 0 )
      goto LABEL_32;
    goto LABEL_16;
  }
  *(_QWORD *)(a4 + 32) = v31 + 1;
  *v31 = 120;
  v14 = *(_DWORD *)(a1 + 148);
  v15 = *(_BYTE **)(a4 + 32);
  v16 = *(_QWORD *)(a4 + 24);
  if ( v14 < 0 )
    goto LABEL_49;
LABEL_13:
  if ( (v14 & 0x40000000) == 0 )
  {
    if ( v16 <= (unsigned __int64)v15 )
    {
      sub_CB5D20(a4, 121);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v15 + 1;
      *v15 = 121;
    }
    goto LABEL_31;
  }
  if ( v16 <= (unsigned __int64)v15 )
  {
    sub_CB5D20(a4, 114);
    goto LABEL_31;
  }
  *(_QWORD *)(a4 + 32) = v15 + 1;
  *v15 = 114;
  v17 = *(_DWORD *)(a1 + 148);
  if ( (v17 & 0x800) == 0 )
    goto LABEL_32;
LABEL_16:
  v18 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v18 < *(_QWORD *)(a4 + 24) )
  {
    *(_QWORD *)(a4 + 32) = v18 + 1;
    *v18 = 110;
    v17 = *(_DWORD *)(a1 + 148);
    if ( (v17 & 0x10000000) == 0 )
      goto LABEL_33;
    goto LABEL_18;
  }
  sub_CB5D20(a4, 110);
  v17 = *(_DWORD *)(a1 + 148);
LABEL_32:
  if ( (v17 & 0x10000000) == 0 )
    goto LABEL_33;
LABEL_18:
  v19 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v19 >= *(_QWORD *)(a4 + 24) )
  {
    sub_CB5D20(a4, 115);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v19 + 1;
    *v19 = 115;
  }
  v17 = *(_DWORD *)(a1 + 148);
LABEL_33:
  if ( (v17 & 0x2000000) != 0 )
  {
    v26 = *(_QWORD *)(a1 + 128);
    if ( *(_QWORD *)(a1 + 136) <= 5u || *(_DWORD *)v26 != 1650811950 || *(_WORD *)(v26 + 4) != 26485 )
    {
      v27 = *(_BYTE **)(a4 + 32);
      if ( (unsigned __int64)v27 >= *(_QWORD *)(a4 + 24) )
      {
        sub_CB5D20(a4, 68);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v27 + 1;
        *v27 = 68;
      }
      v17 = *(_DWORD *)(a1 + 148);
    }
  }
  if ( (v17 & 0x200) != 0 )
  {
    v35 = *(_BYTE **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v35 )
    {
      sub_CB5D20(a4, 105);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v35 + 1;
      *v35 = 105;
    }
  }
  v28 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v28 >= *(_QWORD *)(a4 + 24) )
  {
    sub_CB5D20(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v28 + 1;
    *v28 = 34;
  }
  result = *(_BYTE **)(a4 + 32);
  if ( (*(_BYTE *)(a1 + 149) & 0x10) != 0 )
  {
    v29 = *(_BYTE **)(a4 + 24);
    if ( *(_QWORD *)(a1 + 160) )
    {
      if ( v29 == result )
      {
        sub_CB6200(a4, (unsigned __int8 *)",", 1u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        *result = 44;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 1LL);
        *(_QWORD *)(a4 + 32) = result;
      }
    }
    else if ( (unsigned __int64)(v29 - result) <= 0xB )
    {
      sub_CB6200(a4, "\n\t.linkonce\t", 0xCu);
      result = *(_BYTE **)(a4 + 32);
    }
    else
    {
      qmemcpy(result, "\n\t.linkonce\t", 12);
      result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 12LL);
      *(_QWORD *)(a4 + 32) = result;
    }
    v30 = *(_QWORD *)(a4 + 24);
    switch ( *(_DWORD *)(a1 + 168) )
    {
      case 1:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 7 )
        {
          sub_CB6200(a4, "one_only", 8u);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          *(_QWORD *)result = 0x796C6E6F5F656E6FLL;
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 8LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      case 2:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 6 )
        {
          sub_CB6200(a4, (unsigned __int8 *)"discard", 7u);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          *(_DWORD *)result = 1668508004;
          *((_WORD *)result + 2) = 29281;
          result[6] = 100;
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 7LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      case 3:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 8 )
        {
          sub_CB6200(a4, (unsigned __int8 *)"same_size", 9u);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          result[8] = 101;
          *(_QWORD *)result = 0x7A69735F656D6173LL;
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 9LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      case 4:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 0xC )
        {
          sub_CB6200(a4, "same_contents", 0xDu);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          qmemcpy(result, "same_contents", 13);
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 13LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      case 5:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 0xA )
        {
          sub_CB6200(a4, "associative", 0xBu);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          qmemcpy(result, "associative", 11);
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 11LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      case 6:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 6 )
        {
          sub_CB6200(a4, "largest", 7u);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          *(_DWORD *)result = 1735549292;
          *((_WORD *)result + 2) = 29541;
          result[6] = 116;
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 7LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      case 7:
        if ( (unsigned __int64)(v30 - (_QWORD)result) <= 5 )
        {
          sub_CB6200(a4, "newest", 6u);
          result = *(_BYTE **)(a4 + 32);
        }
        else
        {
          *(_DWORD *)result = 1702323566;
          *((_WORD *)result + 2) = 29811;
          result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 6LL);
          *(_QWORD *)(a4 + 32) = result;
        }
        break;
      default:
        break;
    }
    if ( *(_QWORD *)(a1 + 160) )
    {
      if ( *(_BYTE **)(a4 + 24) == result )
      {
        sub_CB6200(a4, (unsigned __int8 *)",", 1u);
      }
      else
      {
        *result = 44;
        ++*(_QWORD *)(a4 + 32);
      }
      sub_EA12C0(*(_QWORD *)(a1 + 160), a4, a2);
      result = *(_BYTE **)(a4 + 32);
    }
  }
  if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 24) )
    return (_BYTE *)sub_CB5D20(a4, 10);
  *(_QWORD *)(a4 + 32) = result + 1;
  *result = 10;
  return result;
}
