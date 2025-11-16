// Function: sub_BE2390
// Address: 0xbe2390
//
unsigned __int64 __fastcall sub_BE2390(__int64 a1, __int64 a2)
{
  unsigned __int8 v4; // cl
  __int64 v5; // r14
  _QWORD *v6; // rdx
  unsigned __int8 *v7; // rsi
  unsigned __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int8 *v10; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  const char *v13; // r14
  const char *v14; // rax
  unsigned __int64 result; // rax
  unsigned __int8 *v16; // rsi
  _BYTE *v17; // rsi
  unsigned __int8 *v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  const char *v21; // rax
  int v22; // ebx
  __int64 v23; // rdx
  unsigned __int8 v24; // al
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // rdx
  const char *v28; // rax
  const char *v29; // rdx
  _BYTE *v30; // rax
  __int64 v31; // r14
  _BYTE *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rax
  int v36; // edx
  const char *v37; // rax
  __int64 *v38; // rax
  const char *v39; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v40[4]; // [rsp+10h] [rbp-50h] BYREF
  char v41; // [rsp+30h] [rbp-30h]
  char v42; // [rsp+31h] [rbp-2Fh]

  sub_BDAAE0(a1, (const char *)a2);
  if ( (unsigned __int16)sub_AF18C0(a2) != 1
    && (unsigned __int16)sub_AF18C0(a2) != 19
    && (unsigned __int16)sub_AF18C0(a2) != 23
    && (unsigned __int16)sub_AF18C0(a2) != 4
    && (unsigned __int16)sub_AF18C0(a2) != 2
    && (unsigned __int16)sub_AF18C0(a2) != 51
    && (unsigned __int16)sub_AF18C0(a2) != 43 )
  {
    v39 = (const char *)a2;
    v21 = "invalid tag";
    v42 = 1;
LABEL_71:
    v40[0] = v21;
    v41 = 3;
    return sub_BE22A0((_BYTE *)a1, (__int64)v40, &v39);
  }
  v4 = *(_BYTE *)(a2 - 16);
  v5 = a2 - 16;
  if ( (v4 & 2) == 0 )
  {
    v6 = (_QWORD *)(v5 - 8LL * ((v4 >> 2) & 0xF));
    v7 = (unsigned __int8 *)v6[1];
    if ( !v7 )
      goto LABEL_8;
    v8 = *v7;
    if ( (unsigned __int8)v8 <= 0x24u )
      goto LABEL_5;
LABEL_18:
    v13 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1);
    v42 = 1;
    v14 = "invalid scope";
    goto LABEL_12;
  }
  v6 = *(_QWORD **)(a2 - 32);
  v16 = (unsigned __int8 *)v6[1];
  if ( !v16 )
    goto LABEL_8;
  v8 = *v16;
  if ( (unsigned __int8)v8 > 0x24u )
    goto LABEL_18;
LABEL_5:
  v9 = 0x16007FF000LL;
  if ( !_bittest64(&v9, v8) )
    goto LABEL_18;
  if ( (v4 & 2) != 0 )
    v6 = *(_QWORD **)(a2 - 32);
  else
    v6 = (_QWORD *)(v5 - 8LL * ((v4 >> 2) & 0xF));
LABEL_8:
  v10 = (unsigned __int8 *)v6[3];
  if ( v10 )
  {
    v11 = *v10;
    if ( (unsigned __int8)v11 > 0x24u || (v12 = 0x140000F000LL, !_bittest64(&v12, v11)) )
    {
      v13 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 3);
      v42 = 1;
      v14 = "invalid base type";
      goto LABEL_12;
    }
    if ( (v4 & 2) != 0 )
      v6 = *(_QWORD **)(a2 - 32);
    else
      v6 = (_QWORD *)(v5 - 8LL * ((v4 >> 2) & 0xF));
  }
  v17 = (_BYTE *)v6[4];
  if ( v17 )
  {
    if ( *v17 != 5 )
    {
      v13 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4);
      v42 = 1;
      v14 = "invalid composite elements";
LABEL_12:
      v40[0] = v14;
      v41 = 3;
      result = sub_BDD6D0((__int64 *)a1, (__int64)v40);
      if ( *(_QWORD *)a1 )
      {
        result = (unsigned __int64)sub_BD9900((__int64 *)a1, (const char *)a2);
        if ( v13 )
          return (unsigned __int64)sub_BD9900((__int64 *)a1, v13);
      }
      return result;
    }
    if ( (v4 & 2) != 0 )
      v6 = *(_QWORD **)(a2 - 32);
    else
      v6 = (_QWORD *)(v5 - 8LL * ((v4 >> 2) & 0xF));
  }
  v18 = (unsigned __int8 *)v6[5];
  if ( v18 )
  {
    v19 = *v18;
    if ( (unsigned __int8)v19 > 0x24u || (v20 = 0x140000F000LL, !_bittest64(&v20, v19)) )
    {
      v13 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 5);
      v42 = 1;
      v14 = "invalid vtable holder";
      goto LABEL_12;
    }
  }
  v22 = *(_DWORD *)(a2 + 20);
  if ( (v22 & 0x6000) == 0x6000 || (v22 & 0xC00000) == 0xC00000 )
  {
    v42 = 1;
    v40[0] = "invalid reference flags";
    v41 = 3;
    result = sub_BDD6D0((__int64 *)a1, (__int64)v40);
    if ( *(_QWORD *)a1 )
      return (unsigned __int64)sub_BD9900((__int64 *)a1, (const char *)a2);
    return result;
  }
  if ( (v22 & 0x10) != 0 )
  {
    v42 = 1;
    v28 = "DIBlockByRefStruct on DICompositeType is no longer supported";
    goto LABEL_73;
  }
  v23 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4);
  if ( v23 )
  {
    v24 = *(_BYTE *)(v23 - 16);
    if ( (v24 & 2) != 0 )
    {
      v26 = *(_QWORD **)(v23 - 32);
      v27 = (__int64)&v26[*(unsigned int *)(v23 - 24)];
    }
    else
    {
      v25 = (v24 >> 2) & 0xF;
      v26 = (_QWORD *)(v23 - 16 - 8 * v25);
      v27 = v23 - 16 + 8 * (((*(_WORD *)(v23 - 16) >> 6) & 0xF) - v25);
    }
    if ( (_QWORD *)v27 != v26 )
    {
      while ( *v26 )
      {
        if ( (_QWORD *)v27 == ++v26 )
          goto LABEL_54;
      }
      if ( (_QWORD *)v27 != v26 )
      {
        v42 = 1;
        v28 = "DISubprogram contains null entry in `elements` field";
        goto LABEL_73;
      }
    }
  }
LABEL_54:
  if ( (v22 & 0x800) != 0 )
  {
    v35 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4);
    if ( !v35
      || ((*(_BYTE *)(v35 - 16) & 2) == 0 ? (v36 = (*(_WORD *)(v35 - 16) >> 6) & 0xF) : (v36 = *(_DWORD *)(v35 - 24)),
          v36 != 1 || (v38 = (__int64 *)sub_A17150((_BYTE *)(v35 - 16)), (unsigned __int16)sub_AF18C0(*v38) != 33)) )
    {
      v42 = 1;
      v28 = "invalid vector, expected one element of type subrange";
LABEL_73:
      v31 = *(_QWORD *)a1;
      v40[0] = v28;
      v41 = 3;
      if ( v31 )
      {
        sub_CA0E80(v40, v31);
        v32 = *(_BYTE **)(v31 + 32);
        if ( (unsigned __int64)v32 >= *(_QWORD *)(v31 + 24) )
        {
          sub_CB5D20(v31, 10);
        }
        else
        {
          *(_QWORD *)(v31 + 32) = v32 + 1;
          *v32 = 10;
        }
        v33 = *(_QWORD *)a1;
        result = *(unsigned __int8 *)(a1 + 154);
        *(_BYTE *)(a1 + 153) = 1;
        *(_BYTE *)(a1 + 152) |= result;
        if ( v33 )
        {
          sub_A62C00((const char *)a2, v33, a1 + 16, *(_QWORD *)(a1 + 8));
          v34 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            return sub_CB5D20(v34, 10);
          }
          else
          {
            *(_QWORD *)(v34 + 32) = result + 1;
            *(_BYTE *)result = 10;
          }
        }
      }
      else
      {
        result = *(unsigned __int8 *)(a1 + 154);
        *(_BYTE *)(a1 + 153) = 1;
        *(_BYTE *)(a1 + 152) |= result;
      }
      return result;
    }
  }
  v29 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6);
  if ( v29 )
    sub_BDB420(a1, (const char *)a2, v29);
  v30 = (_BYTE *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 8);
  if ( v30 && (*v30 != 13 || (unsigned __int16)sub_AF18C0(a2) != 51) )
  {
    v42 = 1;
    v37 = "discriminator can only appear on variant part";
LABEL_88:
    v40[0] = v37;
    v41 = 3;
    return sub_BDD6D0((__int64 *)a1, (__int64)v40);
  }
  if ( *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 9) && (unsigned __int16)sub_AF18C0(a2) != 1 )
  {
    v42 = 1;
    v37 = "dataLocation can only appear in array type";
    goto LABEL_88;
  }
  if ( *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 10) && (unsigned __int16)sub_AF18C0(a2) != 1 )
  {
    v42 = 1;
    v37 = "associated can only appear in array type";
    goto LABEL_88;
  }
  if ( *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 11) && (unsigned __int16)sub_AF18C0(a2) != 1 )
  {
    v42 = 1;
    v37 = "allocated can only appear in array type";
    goto LABEL_88;
  }
  if ( *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 12) && (unsigned __int16)sub_AF18C0(a2) != 1 )
  {
    v42 = 1;
    v37 = "rank can only appear in array type";
    goto LABEL_88;
  }
  result = sub_AF18C0(a2);
  if ( (_WORD)result == 1 )
  {
    result = (unsigned __int64)sub_A17150((_BYTE *)(a2 - 16));
    if ( !*(_QWORD *)(result + 24) )
    {
      v39 = (const char *)a2;
      v21 = "array types must have a base type";
      v42 = 1;
      goto LABEL_71;
    }
  }
  return result;
}
