// Function: sub_819580
// Address: 0x819580
//
__int64 __fastcall sub_819580(__int64 a1, _QWORD *a2, int a3)
{
  int v4; // r14d
  _BYTE *v5; // rax
  char *v6; // rbx
  int v7; // edx
  int v8; // ecx
  __int64 v9; // r15
  int v10; // r12d
  unsigned __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rsi
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  __int64 result; // rax
  _BYTE *v22; // rdx
  int v23; // [rsp+Ch] [rbp-34h]

  v4 = a3 == 0 ? 34 : 39;
  if ( a2 )
  {
    v5 = (_BYTE *)(*a2)++;
    *v5 = v4;
  }
  v6 = *(char **)(a1 + 16);
  v7 = 1;
  v8 = 0;
  v9 = 1;
  while ( 1 )
  {
    v10 = (unsigned __int8)*v6;
    if ( !(_BYTE)v10 )
      break;
    if ( !v7 )
      goto LABEL_19;
    if ( (_BYTE)v10 == 34 || (_BYTE)v10 == 39 )
      goto LABEL_13;
    if ( (unsigned __int8)(v10 - 76) <= 0x29u )
    {
      v12 = 0x20000000241LL;
      if ( _bittest64(&v12, (unsigned int)(v10 - 76)) )
      {
        v23 = v8;
        v13 = sub_7B7F70(v6);
        v8 = v23;
        if ( v13 != -1 )
          goto LABEL_13;
      }
    }
    else
    {
LABEL_19:
      if ( (_BYTE)v10 == 32 )
        goto LABEL_21;
    }
    v7 = 0;
    if ( !v8 )
      goto LABEL_21;
LABEL_13:
    if ( (_BYTE)v4 != (_BYTE)v10 && (_BYTE)v10 != 92 )
    {
      v7 = 0;
      v8 = 1;
LABEL_21:
      ++v9;
      if ( a2 )
        goto LABEL_17;
      goto LABEL_18;
    }
    v9 += 2;
    if ( a2 )
    {
      v14 = (_BYTE *)*a2;
      v8 = 1;
      ++*a2;
      v7 = 0;
      *v14 = 92;
LABEL_17:
      v15 = (_BYTE *)(*a2)++;
      *v15 = v10;
      goto LABEL_18;
    }
    v7 = 0;
    v8 = 1;
LABEL_18:
    ++v6;
  }
  v11 = (unsigned __int8)v6[1];
  if ( (unsigned __int8)v11 > 0xDu )
    goto LABEL_6;
  v16 = 9648;
  if ( _bittest64(&v16, v11) )
  {
    ++v6;
    v7 = 1;
    v8 = 0;
    goto LABEL_18;
  }
  if ( (_BYTE)v11 != 3 )
  {
    if ( (_BYTE)v11 != 6 )
LABEL_6:
      sub_721090();
    if ( v8 )
    {
      v9 += 4;
      if ( a2 )
      {
        v17 = (_BYTE *)(*a2)++;
        *v17 = 92;
        v18 = (_BYTE *)(*a2)++;
        *v18 = 48;
        v19 = (_BYTE *)(*a2)++;
        *v19 = 48;
        v20 = (_BYTE *)(*a2)++;
        *v20 = 48;
      }
    }
    ++v6;
    goto LABEL_18;
  }
  result = v9 + 1;
  if ( a2 )
  {
    v22 = (_BYTE *)(*a2)++;
    *v22 = v4;
  }
  return result;
}
