// Function: sub_38D7C30
// Address: 0x38d7c30
//
_BYTE *__fastcall sub_38D7C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v7; // rdx
  __int64 v8; // r14
  _WORD *v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  char *v12; // rsi
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
  char *v24; // rsi
  _BYTE *result; // rax
  _BYTE *v26; // rax
  _BYTE *v27; // rax
  _BYTE *v28; // rdx
  __int64 v29; // rcx
  _BYTE *v30; // rax
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax

  if ( sub_38D7BC0(a1, *(_QWORD *)(a1 + 152), *(_QWORD *)(a1 + 160)) )
  {
    v20 = *(_BYTE **)(a4 + 24);
    if ( (unsigned __int64)v20 >= *(_QWORD *)(a4 + 16) )
    {
      a4 = sub_16E7DE0(a4, 9);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v20 + 1;
      *v20 = 9;
    }
    v21 = *(_QWORD *)(a4 + 16);
    v22 = *(_BYTE **)(a4 + 24);
    v23 = *(_QWORD *)(a1 + 160);
    v24 = *(char **)(a1 + 152);
    if ( v23 > v21 - (unsigned __int64)v22 )
    {
      v36 = sub_16E7EE0(a4, v24, *(_QWORD *)(a1 + 160));
      v22 = *(_BYTE **)(v36 + 24);
      a4 = v36;
      v21 = *(_QWORD *)(v36 + 16);
    }
    else if ( v23 )
    {
      memcpy(v22, v24, *(_QWORD *)(a1 + 160));
      v21 = *(_QWORD *)(a4 + 16);
      v22 = (_BYTE *)(v23 + *(_QWORD *)(a4 + 24));
      *(_QWORD *)(a4 + 24) = v22;
    }
    if ( v21 > (unsigned __int64)v22 )
    {
      *(_QWORD *)(a4 + 24) = v22 + 1;
      *v22 = 10;
      return v22 + 1;
    }
    return (_BYTE *)sub_16E7DE0(a4, 10);
  }
  v7 = *(void **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v7 <= 9u )
  {
    v10 = sub_16E7EE0(a4, "\t.section\t", 0xAu);
    v9 = *(_WORD **)(v10 + 24);
    v8 = v10;
  }
  else
  {
    v8 = a4;
    qmemcpy(v7, "\t.section\t", 10);
    v9 = (_WORD *)(*(_QWORD *)(a4 + 24) + 10LL);
    *(_QWORD *)(a4 + 24) = v9;
  }
  v11 = *(_QWORD *)(a1 + 160);
  v12 = *(char **)(a1 + 152);
  v13 = *(_QWORD *)(v8 + 16) - (_QWORD)v9;
  if ( v11 > v13 )
  {
    v33 = sub_16E7EE0(v8, v12, *(_QWORD *)(a1 + 160));
    v9 = *(_WORD **)(v33 + 24);
    v8 = v33;
    v13 = *(_QWORD *)(v33 + 16) - (_QWORD)v9;
  }
  else if ( v11 )
  {
    memcpy(v9, v12, *(_QWORD *)(a1 + 160));
    v35 = *(_QWORD *)(v8 + 16);
    v9 = (_WORD *)(v11 + *(_QWORD *)(v8 + 24));
    *(_QWORD *)(v8 + 24) = v9;
    v13 = v35 - (_QWORD)v9;
  }
  if ( v13 <= 1 )
  {
    sub_16E7EE0(v8, ",\"", 2u);
    v14 = *(_DWORD *)(a1 + 168);
    if ( (v14 & 0x40) == 0 )
      goto LABEL_10;
  }
  else
  {
    *v9 = 8748;
    *(_QWORD *)(v8 + 24) += 2LL;
    v14 = *(_DWORD *)(a1 + 168);
    if ( (v14 & 0x40) == 0 )
      goto LABEL_10;
  }
  v31 = *(_BYTE **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v31 )
  {
    *(_QWORD *)(a4 + 24) = v31 + 1;
    *v31 = 100;
    v14 = *(_DWORD *)(a1 + 168);
    if ( (v14 & 0x80u) == 0 )
      goto LABEL_11;
    goto LABEL_51;
  }
  sub_16E7DE0(a4, 100);
  v14 = *(_DWORD *)(a1 + 168);
LABEL_10:
  if ( (v14 & 0x80u) == 0 )
    goto LABEL_11;
LABEL_51:
  v32 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)v32 >= *(_QWORD *)(a4 + 16) )
  {
    sub_16E7DE0(a4, 98);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v32 + 1;
    *v32 = 98;
  }
  v14 = *(_DWORD *)(a1 + 168);
LABEL_11:
  if ( (v14 & 0x20000000) == 0 )
    goto LABEL_12;
  v30 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)v30 >= *(_QWORD *)(a4 + 16) )
  {
    sub_16E7DE0(a4, 120);
    v14 = *(_DWORD *)(a1 + 168);
LABEL_12:
    v15 = *(_BYTE **)(a4 + 24);
    v16 = *(_QWORD *)(a4 + 16);
    if ( v14 >= 0 )
      goto LABEL_13;
LABEL_46:
    if ( (unsigned __int64)v15 >= v16 )
    {
      sub_16E7DE0(a4, 119);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v15 + 1;
      *v15 = 119;
    }
LABEL_31:
    v17 = *(_DWORD *)(a1 + 168);
    if ( (v17 & 0x800) == 0 )
      goto LABEL_32;
    goto LABEL_16;
  }
  *(_QWORD *)(a4 + 24) = v30 + 1;
  *v30 = 120;
  v14 = *(_DWORD *)(a1 + 168);
  v15 = *(_BYTE **)(a4 + 24);
  v16 = *(_QWORD *)(a4 + 16);
  if ( v14 < 0 )
    goto LABEL_46;
LABEL_13:
  if ( (v14 & 0x40000000) == 0 )
  {
    if ( (unsigned __int64)v15 >= v16 )
    {
      sub_16E7DE0(a4, 121);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v15 + 1;
      *v15 = 121;
    }
    goto LABEL_31;
  }
  if ( (unsigned __int64)v15 >= v16 )
  {
    sub_16E7DE0(a4, 114);
    goto LABEL_31;
  }
  *(_QWORD *)(a4 + 24) = v15 + 1;
  *v15 = 114;
  v17 = *(_DWORD *)(a1 + 168);
  if ( (v17 & 0x800) == 0 )
    goto LABEL_32;
LABEL_16:
  v18 = *(_BYTE **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v18 )
  {
    *(_QWORD *)(a4 + 24) = v18 + 1;
    *v18 = 110;
    v17 = *(_DWORD *)(a1 + 168);
    if ( (v17 & 0x10000000) == 0 )
      goto LABEL_33;
    goto LABEL_18;
  }
  sub_16E7DE0(a4, 110);
  v17 = *(_DWORD *)(a1 + 168);
LABEL_32:
  if ( (v17 & 0x10000000) == 0 )
    goto LABEL_33;
LABEL_18:
  v19 = *(_BYTE **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v19 )
  {
    sub_16E7DE0(a4, 115);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v19 + 1;
    *v19 = 115;
  }
  v17 = *(_DWORD *)(a1 + 168);
LABEL_33:
  if ( (v17 & 0x2000000) != 0 )
  {
    if ( *(_QWORD *)(a1 + 160) <= 5u
      || (v34 = *(_QWORD *)(a1 + 152), *(_DWORD *)v34 != 1650811950)
      || *(_WORD *)(v34 + 4) != 26485 )
    {
      v26 = *(_BYTE **)(a4 + 24);
      if ( (unsigned __int64)v26 >= *(_QWORD *)(a4 + 16) )
      {
        sub_16E7DE0(a4, 68);
      }
      else
      {
        *(_QWORD *)(a4 + 24) = v26 + 1;
        *v26 = 68;
      }
    }
  }
  v27 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)v27 >= *(_QWORD *)(a4 + 16) )
  {
    sub_16E7DE0(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v27 + 1;
    *v27 = 34;
  }
  result = *(_BYTE **)(a4 + 24);
  if ( (*(_BYTE *)(a1 + 169) & 0x10) != 0 )
  {
    v28 = *(_BYTE **)(a4 + 16);
    if ( *(_QWORD *)(a1 + 176) )
    {
      if ( result == v28 )
      {
        sub_16E7EE0(a4, ",", 1u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        *result = 44;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 1LL);
        *(_QWORD *)(a4 + 24) = result;
      }
    }
    else if ( (unsigned __int64)(v28 - result) <= 0xB )
    {
      sub_16E7EE0(a4, "\n\t.linkonce\t", 0xCu);
      result = *(_BYTE **)(a4 + 24);
    }
    else
    {
      qmemcpy(result, "\n\t.linkonce\t", 12);
      result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 12LL);
      *(_QWORD *)(a4 + 24) = result;
    }
    v29 = *(_QWORD *)(a4 + 16);
    switch ( *(_DWORD *)(a1 + 184) )
    {
      case 1:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 7 )
        {
          sub_16E7EE0(a4, "one_only", 8u);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          *(_QWORD *)result = 0x796C6E6F5F656E6FLL;
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 8LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      case 2:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 6 )
        {
          sub_16E7EE0(a4, "discard", 7u);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          *(_DWORD *)result = 1668508004;
          *((_WORD *)result + 2) = 29281;
          result[6] = 100;
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 7LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      case 3:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 8 )
        {
          sub_16E7EE0(a4, "same_size", 9u);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          result[8] = 101;
          *(_QWORD *)result = 0x7A69735F656D6173LL;
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 9LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      case 4:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 0xC )
        {
          sub_16E7EE0(a4, "same_contents", 0xDu);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          qmemcpy(result, "same_contents", 13);
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 13LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      case 5:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 0xA )
        {
          sub_16E7EE0(a4, "associative", 0xBu);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          qmemcpy(result, "associative", 11);
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 11LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      case 6:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 6 )
        {
          sub_16E7EE0(a4, "largest", 7u);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          *(_DWORD *)result = 1735549292;
          *((_WORD *)result + 2) = 29541;
          result[6] = 116;
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 7LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      case 7:
        if ( (unsigned __int64)(v29 - (_QWORD)result) <= 5 )
        {
          sub_16E7EE0(a4, "newest", 6u);
          result = *(_BYTE **)(a4 + 24);
        }
        else
        {
          *(_DWORD *)result = 1702323566;
          *((_WORD *)result + 2) = 29811;
          result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 6LL);
          *(_QWORD *)(a4 + 24) = result;
        }
        break;
      default:
        break;
    }
    if ( *(_QWORD *)(a1 + 176) )
    {
      if ( *(_BYTE **)(a4 + 16) == result )
      {
        sub_16E7EE0(a4, ",", 1u);
      }
      else
      {
        *result = 44;
        ++*(_QWORD *)(a4 + 24);
      }
      sub_38E2490(*(_QWORD *)(a1 + 176), a4, a2);
      result = *(_BYTE **)(a4 + 24);
    }
  }
  if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 16) )
    return (_BYTE *)sub_16E7DE0(a4, 10);
  *(_QWORD *)(a4 + 24) = result + 1;
  *result = 10;
  return result;
}
