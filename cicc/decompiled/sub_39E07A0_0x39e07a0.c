// Function: sub_39E07A0
// Address: 0x39e07a0
//
__int64 __fastcall sub_39E07A0(__int64 *a1, __int64 a2, unsigned int a3)
{
  char *v5; // r13
  __int64 v6; // rbx
  size_t v7; // r12
  void *v8; // rdi
  __int64 v9; // rdi
  __int64 result; // rax
  int i; // eax
  unsigned __int64 v12; // rcx
  unsigned int v13; // r13d
  int v14; // r14d
  unsigned __int64 v15; // r11
  unsigned int *v16; // rax
  char v17; // [rsp+Fh] [rbp-41h]
  _QWORD v18[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a3 == 4 )
  {
    v5 = *(char **)(a1[35] + 216);
    if ( !v5 )
      goto LABEL_13;
    goto LABEL_6;
  }
  if ( a3 <= 4 )
  {
    if ( a3 != 1 )
    {
      if ( a3 == 2 )
      {
        v5 = *(char **)(a1[35] + 208);
        if ( v5 )
          goto LABEL_6;
      }
LABEL_13:
      if ( sub_38CF290(a2, v18) )
      {
        result = *(unsigned __int8 *)(a1[35] + 16);
        v17 = *(_BYTE *)(a1[35] + 16);
        if ( !a3 )
          return result;
        goto LABEL_15;
      }
LABEL_34:
      sub_16BD130("Don't know how to emit this value.", 1u);
    }
    v5 = *(char **)(a1[35] + 200);
    if ( !v5 )
      goto LABEL_13;
LABEL_6:
    v6 = a1[34];
    v7 = strlen(v5);
    v8 = *(void **)(v6 + 24);
    if ( v7 > *(_QWORD *)(v6 + 16) - (_QWORD)v8 )
    {
      sub_16E7EE0(v6, v5, v7);
    }
    else if ( v7 )
    {
      memcpy(v8, v5, v7);
      *(_QWORD *)(v6 + 24) += v7;
    }
    v9 = a1[2];
    if ( v9 )
      return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 56LL))(v9, a2);
    sub_38CDBE0(a2, a1[34], a1[35]);
    return (__int64)sub_39E06C0((__int64)a1);
  }
  if ( a3 == 8 )
  {
    v5 = *(char **)(a1[35] + 224);
    if ( !v5 )
      goto LABEL_13;
    goto LABEL_6;
  }
  if ( !sub_38CF290(a2, v18) )
    goto LABEL_34;
  v17 = *(_BYTE *)(a1[35] + 16);
LABEL_15:
  for ( i = 0; ; i = v14 )
  {
    v13 = a3 - 1;
    if ( a3 - i <= a3 - 1 )
      v13 = a3 - i;
    if ( v13 )
    {
      _BitScanReverse64(&v12, v13);
      v13 = 0x8000000000000000LL >> ((unsigned __int8)v12 ^ 0x3Fu);
      v14 = i + v13;
      v15 = 0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - (unsigned __int8)v13));
    }
    else
    {
      v14 = i;
      v15 = 0;
    }
    if ( !v17 )
      LOBYTE(i) = a3 - i - v13;
    v16 = (unsigned int *)sub_38CB470(v15 & (v18[0] >> (8 * (unsigned __int8)i)), a1[1]);
    result = sub_38DDD30((__int64)a1, v16);
    if ( a3 == v14 )
      break;
  }
  return result;
}
