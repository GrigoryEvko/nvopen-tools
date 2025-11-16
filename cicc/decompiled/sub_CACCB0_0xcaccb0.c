// Function: sub_CACCB0
// Address: 0xcaccb0
//
__int64 __fastcall sub_CACCB0(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // r13d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  char *v10; // r8
  __int64 v11; // r9
  char *v12; // rsi
  char v13; // al
  char v14; // si
  char v15; // si
  char *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  _QWORD *v20; // rdi
  char v21; // si
  __int64 v22; // r9
  _BYTE *v23; // rsi
  __int64 v24; // r9
  char v25; // si
  _BYTE *v26; // rsi
  _BYTE *v27; // rsi
  unsigned __int64 v28; // r14
  __int64 v29; // rbx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  char v32; // si
  _QWORD v33[2]; // [rsp+0h] [rbp-60h] BYREF
  const char *v34; // [rsp+10h] [rbp-50h] BYREF
  char v35; // [rsp+30h] [rbp-30h]
  char v36; // [rsp+31h] [rbp-2Fh]

  v3 = *(unsigned __int8 *)(a1 + 72);
  if ( (_BYTE)v3 )
    return sub_CAA420(a1);
  sub_CA8400(a1);
  if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)(a1 + 48) )
    return sub_CAA6A0(a1);
  sub_CA81D0((__int64 *)a1, a2, v5, v6, v7);
  sub_CA9FC0(a1, *(_DWORD *)(a1 + 60));
  v12 = *(char **)(a1 + 40);
  v13 = *v12;
  if ( !*(_DWORD *)(a1 + 60) )
  {
    if ( v13 == 37 )
      return sub_CAC150(a1);
    v8 = *(_QWORD *)(a1 + 48);
    v9 = (__int64)(v12 + 4);
    if ( v8 >= (unsigned __int64)(v12 + 4) )
    {
      if ( v13 != 45 )
        goto LABEL_8;
      if ( v12[1] != 45 || v12[2] != 45 )
        goto LABEL_45;
      v26 = v12 + 3;
      if ( (_BYTE *)v8 == v26 || (unsigned __int8)sub_CA7F80(a1, v26) )
      {
        v32 = 1;
        return sub_CAA2B0(a1, v32);
      }
      v12 = *(char **)(a1 + 40);
      v9 = *(unsigned int *)(a1 + 60);
      v13 = *v12;
      if ( !(_DWORD)v9 )
      {
        v8 = *(_QWORD *)(a1 + 48);
        v9 = (__int64)(v12 + 4);
        if ( v8 >= (unsigned __int64)(v12 + 4) )
        {
LABEL_8:
          if ( v13 != 46 )
            goto LABEL_9;
          if ( v12[1] != 46 || v12[2] != 46 )
            goto LABEL_34;
          v27 = v12 + 3;
          if ( v27 != (_BYTE *)v8 && !(unsigned __int8)sub_CA7F80(a1, v27) )
          {
            v12 = *(char **)(a1 + 40);
            v13 = *v12;
            goto LABEL_9;
          }
          v32 = 0;
          return sub_CAA2B0(a1, v32);
        }
      }
    }
  }
LABEL_9:
  switch ( v13 )
  {
    case '[':
      v15 = 1;
      return sub_CAA9B0(a1, v15);
    case '{':
      v15 = 0;
      return sub_CAA9B0(a1, v15);
    case ']':
      v21 = 1;
      return sub_CAA820(a1, v21);
    case '}':
      v21 = 0;
      return sub_CAA820(a1, v21);
    case ',':
      return sub_CAA150(a1);
  }
  if ( v13 != 45 )
  {
    if ( v13 != 63 )
      goto LABEL_16;
    v10 = v12 + 1;
    if ( v12 + 1 == *(char **)(a1 + 48) )
      return sub_CAB5F0(a1, (__int64)v12, v8, v9, (__int64)v10, v11);
    goto LABEL_31;
  }
LABEL_45:
  if ( (unsigned __int8)sub_CA7F80(a1, ++v12) )
    return sub_CAB480(a1, (__int64)v12, v8, v9, (__int64)v10, v22);
  v12 = *(char **)(a1 + 40);
  v10 = v12 + 1;
  if ( v12 + 1 == *(char **)(a1 + 48) )
    return sub_CAB480(a1, (__int64)v12, v8, v9, (__int64)v10, v22);
  v13 = *v12;
  if ( *v12 == 63 )
  {
LABEL_31:
    v12 = v10;
    if ( !(unsigned __int8)sub_CA7F80(a1, v10) )
    {
      v12 = *(char **)(a1 + 40);
      v13 = *v12;
      goto LABEL_16;
    }
    return sub_CAB5F0(a1, (__int64)v12, v8, v9, (__int64)v10, v11);
  }
LABEL_16:
  if ( v13 == 58 )
  {
    v23 = v12 + 1;
    if ( !(unsigned __int8)sub_CA7FB0(a1, v23) || *(_BYTE *)(a1 + 74) )
      return sub_CAB780(a1, (__int64)v23, v8, v9, (__int64)v10, v24);
    v12 = *(char **)(a1 + 40);
    v13 = *v12;
  }
  switch ( v13 )
  {
    case '*':
      v25 = 1;
      return sub_CAAD30(a1, v25);
    case '&':
      v25 = 0;
      return sub_CAAD30(a1, v25);
    case '!':
      return sub_CAAB30(a1);
    case '|':
      if ( !*(_DWORD *)(a1 + 68) )
        return sub_CAC470(a1);
      break;
    case '>':
      if ( !*(_DWORD *)(a1 + 68) )
        return sub_CAC470(a1);
      break;
    case '\'':
      v14 = 0;
      return sub_CAAF80(a1, v14, v8, v9, (__int64)v10);
    case '"':
      v14 = 1;
      return sub_CAAF80(a1, v14, v8, v9, (__int64)v10);
    default:
      break;
  }
LABEL_34:
  v33[0] = v12;
  v33[1] = 1;
  if ( ((unsigned __int8)sub_CA7F80(a1, v12)
     || (v16 = "-?:,[]{}#&*!|>'\"%@`", sub_C934D0(v33, "-?:,[]{}#&*!|>'\"%@`", 19, 0) != -1))
    && ((v16 = "?:-", v20 = v33, sub_C934D0(v33, "?:-", 3, 0) == -1)
     || (v20 = (_QWORD *)a1, v16 = (char *)(*(_QWORD *)(a1 + 40) + 1LL), !(unsigned __int8)sub_CA7FB0(a1, v16))) )
  {
    v28 = *(_QWORD *)(a1 + 40);
    v29 = *(_QWORD *)(a1 + 336);
    v36 = 1;
    v34 = "Unrecognized character while tokenizing.";
    v30 = *(_QWORD *)(a1 + 48);
    v35 = 3;
    if ( v28 >= v30 )
      v28 = v30 - 1;
    if ( v29 )
    {
      v31 = sub_2241E50(v20, v16, v30 - 1, v18, v19);
      *(_DWORD *)v29 = 22;
      *(_QWORD *)(v29 + 8) = v31;
    }
    if ( !*(_BYTE *)(a1 + 75) )
      sub_C91CB0(*(__int64 **)a1, v28, 0, (__int64)&v34, 0, 0, 0, 0, *(_BYTE *)(a1 + 76));
    *(_BYTE *)(a1 + 75) = 1;
  }
  else
  {
    return (unsigned int)sub_CABD70(a1, v16, v17, v18, v19);
  }
  return v3;
}
