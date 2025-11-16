// Function: sub_A06350
// Address: 0xa06350
//
__int64 __fastcall sub_A06350(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned __int64 v3; // rdx
  _BYTE **v4; // rbx
  _BYTE **v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  _BYTE *v9; // rsi
  __int64 v10; // r12
  __int64 v12; // [rsp+8h] [rbp-148h]
  _BYTE *v13; // [rsp+10h] [rbp-140h] BYREF
  __int64 v14; // [rsp+18h] [rbp-138h]
  _BYTE v15[304]; // [rsp+20h] [rbp-130h] BYREF

  v14 = 0x2000000000LL;
  v2 = *(_BYTE *)(a2 - 16);
  v13 = v15;
  if ( (v2 & 2) != 0 )
  {
    v3 = *(unsigned int *)(a2 - 24);
    if ( v3 <= 0x20 )
      goto LABEL_3;
    sub_C8D5F0(&v13, v15, v3, 8);
    v2 = *(_BYTE *)(a2 - 16);
    if ( (v2 & 2) != 0 )
    {
      v3 = *(unsigned int *)(a2 - 24);
LABEL_3:
      v4 = *(_BYTE ***)(a2 - 32);
      v5 = &v4[v3];
      if ( v5 != v4 )
        goto LABEL_4;
LABEL_12:
      v8 = (unsigned int)v14;
      goto LABEL_7;
    }
  }
  v4 = (_BYTE **)(a2 - 16 - 8LL * ((v2 >> 2) & 0xF));
  v5 = &v4[(*(_WORD *)(a2 - 16) >> 6) & 0xF];
  if ( v5 == v4 )
    goto LABEL_12;
  do
  {
LABEL_4:
    v6 = sub_A05F80(a1, *v4);
    v7 = (unsigned int)v14;
    if ( (unsigned __int64)(unsigned int)v14 + 1 > HIDWORD(v14) )
    {
      v12 = v6;
      sub_C8D5F0(&v13, v15, (unsigned int)v14 + 1LL, 8);
      v7 = (unsigned int)v14;
      v6 = v12;
    }
    ++v4;
    *(_QWORD *)&v13[8 * v7] = v6;
    v8 = (unsigned int)(v14 + 1);
    LODWORD(v14) = v14 + 1;
  }
  while ( v5 != v4 );
LABEL_7:
  v9 = v13;
  v10 = sub_B9C770(*(_QWORD *)(a1 + 216), v13, v8, 0, 1);
  if ( v13 != v15 )
    _libc_free(v13, v9);
  return v10;
}
