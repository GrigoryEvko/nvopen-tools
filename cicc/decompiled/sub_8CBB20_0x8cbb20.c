// Function: sub_8CBB20
// Address: 0x8cbb20
//
__int64 *__fastcall sub_8CBB20(unsigned __int8 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 *v7; // r14
  __int64 v8; // rdi
  unsigned __int8 v9; // r8
  __int64 v10; // rax
  unsigned __int8 v11; // r8
  __int64 *result; // rax
  __int64 v13; // r12
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  unsigned __int8 v20; // [rsp+Ch] [rbp-34h]
  unsigned __int8 v21; // [rsp+Ch] [rbp-34h]

  if ( a1 == 37 )
  {
    v5 = a3[8];
    v6 = (__int64 *)(a2 + 64);
    v7 = a3 + 8;
  }
  else
  {
    v5 = a3[4];
    v6 = (__int64 *)(a2 + 32);
    v7 = a3 + 4;
  }
  v8 = *v6;
  v9 = a1;
  if ( v5 )
  {
    if ( v5 != v8 && v8 )
    {
      if ( *(_DWORD *)(v8 + 16) > 1u && *(_DWORD *)(v5 + 16) == 1 )
      {
        v18 = v6;
        v8 = *v7;
        v6 = v7;
        v7 = v18;
      }
      sub_8D0810(v8);
      v8 = *v6;
      v9 = a1;
    }
LABEL_10:
    v10 = *v7;
    if ( *v7 == v8 )
      goto LABEL_12;
    goto LABEL_11;
  }
  if ( !v8 )
  {
    v19 = sub_8D07C0();
    v9 = a1;
    *v7 = v19;
    *(_BYTE *)(v19 + 20) = a1;
    ++*(_DWORD *)(*v7 + 16);
    *(_QWORD *)*v7 = a3;
    v8 = *v6;
    goto LABEL_10;
  }
  *v7 = v8;
  ++*(_DWORD *)(v8 + 16);
  if ( a1 == 37 )
  {
    v14 = a3[7];
    v15 = *(_QWORD **)(v14 + 32);
    if ( !v15 || v14 == *v15 )
      *(_QWORD *)a3[8] = a3;
    v10 = *v7;
    if ( *v6 == *v7 )
      goto LABEL_38;
  }
  else
  {
    sub_8CB6C0(a1, (__int64)a3);
    v10 = *v7;
    v9 = a1;
    if ( *v6 == *v7 )
      goto LABEL_13;
  }
LABEL_11:
  *v6 = v10;
  ++*(_DWORD *)(v10 + 16);
LABEL_12:
  if ( a1 != 37 )
  {
LABEL_13:
    v20 = v9;
    sub_8CB6C0(v9, a2);
    v11 = v20;
    goto LABEL_14;
  }
LABEL_38:
  v16 = *(_QWORD *)(a2 + 56);
  v17 = *(_QWORD **)(v16 + 32);
  if ( !v17 || v16 == *v17 )
  {
    v11 = 37;
    **(_QWORD **)(a2 + 64) = a2;
  }
  else
  {
    v11 = 37;
  }
LABEL_14:
  result = (__int64 *)*v7;
  if ( (*(_BYTE *)(a3 - 1) & 2) == 0 )
  {
    result[1] = (__int64)a3;
    if ( (*(_BYTE *)(a2 - 8) & 2) == 0 )
      return result;
LABEL_19:
    result = (__int64 *)*v7;
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(a2 - 8) & 2) == 0 )
  {
    result[1] = a2;
    if ( (*(_BYTE *)(a2 - 8) & 2) == 0 )
      return result;
    goto LABEL_19;
  }
LABEL_20:
  v13 = *result;
  if ( a2 == *result || (*(_BYTE *)(v13 - 8) & 2) == 0 )
    return result;
  if ( a1 == 37 )
  {
    if ( *(char *)(a2 - 8) >= 0 )
      return result;
  }
  else
  {
    if ( (*(_BYTE *)(a2 + 88) & 8) != 0 )
    {
      v21 = v11;
      result = (__int64 *)sub_7604D0(*result, v11);
      v11 = v21;
    }
    if ( *(char *)(a2 - 8) >= 0 )
      goto LABEL_26;
  }
  result = (__int64 *)sub_75B260(v13, v11);
LABEL_26:
  if ( a1 == 6 )
  {
    result = (__int64 *)((unsigned int)*(unsigned __int8 *)(a2 + 140) - 9);
    if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 9) <= 2u )
    {
      result = (__int64 *)((unsigned int)*(unsigned __int8 *)(v13 + 140) - 9);
      if ( (unsigned __int8)(*(_BYTE *)(v13 + 140) - 9) <= 2u )
      {
        result = (__int64 *)*(unsigned __int8 *)(a2 + 178);
        if ( ((unsigned __int8)result & 0x40) != 0 )
        {
          sub_75C030(v13);
          result = (__int64 *)*(unsigned __int8 *)(a2 + 178);
        }
        if ( (char)result < 0 )
          return sub_75BF90(v13);
      }
    }
  }
  return result;
}
