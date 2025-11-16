// Function: sub_747370
// Address: 0x747370
//
__int64 __fastcall sub_747370(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64); // rax
  __int64 v10; // r14
  char v11; // al
  __int64 *i; // r13
  char v13; // al
  __int64 v14; // rax
  __int64 *v15; // r15
  unsigned int (__fastcall *v16)(__int64 *); // rax
  unsigned int (__fastcall *v17)(__int64 *); // rax
  char v18; // al
  unsigned int v19; // r13d
  __int64 (__fastcall *v20)(__int64 *); // rax
  bool v21; // r9
  int v22; // edx
  __int64 v23; // rax
  char v24; // dl
  _BOOL4 v25; // eax
  bool v26; // [rsp+8h] [rbp-88h]
  bool v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+10h] [rbp-80h]
  bool v30; // [rsp+1Fh] [rbp-71h]
  _BYTE v31[112]; // [rsp+20h] [rbp-70h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)result == 2 )
  {
    v6 = *(_QWORD *)(a1 + 32);
    v7 = 64;
    v8 = sub_746B90((_DWORD *)(v6 + 128));
    if ( !v8 )
    {
      v8 = v6;
      v7 = 59;
    }
    v9 = *(__int64 (__fastcall **)(__int64, __int64))(a2 + 32);
    if ( v9 )
      result = v9(v8, v7);
    else
      result = sub_74C550(v8, v7, a2);
LABEL_11:
    if ( (*(_BYTE *)(a1 + 24) & 0x30) != 0 )
      return (*(__int64 (__fastcall **)(char *, __int64))a2)("...", a2);
    return result;
  }
  if ( (unsigned __int8)result > 2u )
  {
    if ( (_BYTE)result != 3 )
      sub_721090();
    goto LABEL_11;
  }
  if ( !(_BYTE)result )
  {
    result = sub_74B930(*(_QWORD *)(a1 + 32), a2);
    goto LABEL_11;
  }
  *(_BYTE *)(a2 + 156) = 1;
  if ( (*(_BYTE *)(a1 + 24) & 1) != 0 )
  {
    (*(void (__fastcall **)(const char *))a2)("array-bound=");
    v5 = *(_QWORD *)(a1 + 32);
    if ( v5 > 9 )
    {
      sub_622470(v5, v31);
    }
    else
    {
      v31[1] = 0;
      v31[0] = v5 + 48;
    }
    result = (*(__int64 (__fastcall **)(_BYTE *, __int64))a2)(v31, a2);
    goto LABEL_8;
  }
  v10 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 48) && !v10 )
  {
    result = (*(__int64 (__fastcall **)(const char *))a2)("<expression>");
    goto LABEL_8;
  }
  v29 = *(_QWORD *)(v10 + 144);
  v11 = *(_BYTE *)(v10 + 172);
  v30 = (v11 & 4) != 0;
  if ( *(_BYTE *)(a2 + 159) )
  {
    *(_QWORD *)(v10 + 144) = 0;
    *(_BYTE *)(v10 + 172) = v11 & 0xFB;
  }
  for ( i = sub_72ECB0(v10); ; i = *(__int64 **)(v14 + 144) )
  {
    if ( !i )
    {
      v15 = 0;
      goto LABEL_32;
    }
    v13 = *((_BYTE *)i + 24);
    if ( v13 != 2 )
      break;
    v14 = i[7];
    v15 = *(__int64 **)(v14 + 144);
    if ( !v15 && (*(_BYTE *)(v14 + 172) & 4) == 0 )
      goto LABEL_29;
    if ( (*(_BYTE *)(v14 + 170) & 0x10) != 0 )
      goto LABEL_49;
  }
  v24 = *((_BYTE *)i + 27);
  if ( (v24 & 2) == 0 )
  {
LABEL_49:
    v15 = 0;
    goto LABEL_29;
  }
  v15 = 0;
  if ( v13 == 1 && *((_BYTE *)i + 56) == 5 && (*(_BYTE *)(a1 + 25) & 0xC) != 0 )
  {
    v15 = i;
    *((_BYTE *)i + 27) = v24 & 0xFD;
  }
LABEL_29:
  v16 = *(unsigned int (__fastcall **)(__int64 *))(a2 + 112);
  if ( v16 && v16(i) )
  {
    *(_BYTE *)(v10 + 172) &= ~4u;
    i = 0;
    *(_QWORD *)(v10 + 144) = 0;
  }
  else
  {
    v20 = *(__int64 (__fastcall **)(__int64 *))(a2 + 120);
    if ( v20 )
      i = (__int64 *)v20(i);
  }
LABEL_32:
  if ( !*(_BYTE *)(a2 + 136) )
    goto LABEL_42;
  v17 = *(unsigned int (__fastcall **)(__int64 *))(a2 + 104);
  if ( v17 && !v17(i) )
  {
    if ( (*(_BYTE *)(a1 + 24) & 0x10) == 0 )
      goto LABEL_42;
    if ( i )
    {
      v18 = *((_BYTE *)i + 24);
      if ( v18 != 5 )
      {
        if ( v18 == 2 )
        {
          v19 = *(_BYTE *)(i[7] + 173) != 10;
          goto LABEL_43;
        }
        goto LABEL_39;
      }
LABEL_42:
      v19 = 0;
      goto LABEL_43;
    }
  }
LABEL_39:
  v19 = 1;
LABEL_43:
  if ( (unsigned int)sub_8D32E0(*(_QWORD *)(v10 + 128)) )
  {
    v23 = sub_8D46C0(*(_QWORD *)(v10 + 128));
    if ( (*(_BYTE *)(a1 + 25) & 8) != 0 && ((v28 = v23, (unsigned int)sub_8D2310(v23)) || (unsigned int)sub_8D3410(v28)) )
    {
      (*(void (__fastcall **)(char *, __int64))a2)("(", a2);
      sub_747250(v10, v19, a2);
      (*(void (__fastcall **)(char *, __int64))a2)(")", a2);
    }
    else
    {
      sub_747250(v10, v19, a2);
    }
  }
  else
  {
    v21 = (*(_BYTE *)(v10 + 168) & 8) != 0;
    if ( *(_BYTE *)(v10 + 173) == 7 )
    {
      v27 = (*(_BYTE *)(v10 + 168) & 8) != 0;
      v25 = sub_737660(v10);
      v21 = v27;
      if ( !v25 )
        *(_BYTE *)(v10 + 168) &= ~8u;
    }
    v26 = v21;
    sub_748000(v10, v19, a2);
    *(_BYTE *)(v10 + 168) = *(_BYTE *)(v10 + 168) & 0xF7 | (8 * v26);
  }
  v22 = *(unsigned __int8 *)(v10 + 172);
  *(_QWORD *)(v10 + 144) = v29;
  v22 &= ~4u;
  result = v22 | (4 * (unsigned int)v30);
  *(_BYTE *)(v10 + 172) = v22 | (4 * v30);
  if ( v15 )
    *((_BYTE *)v15 + 27) |= 2u;
LABEL_8:
  *(_BYTE *)(a2 + 156) = 0;
  if ( (*(_BYTE *)(a1 + 24) & 0x30) != 0 )
    return (*(__int64 (__fastcall **)(char *, __int64))a2)("...", a2);
  return result;
}
