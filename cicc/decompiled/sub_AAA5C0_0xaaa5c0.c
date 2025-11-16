// Function: sub_AAA5C0
// Address: 0xaaa5c0
//
__int64 __fastcall sub_AAA5C0(__int64 a1, _BYTE *a2, unsigned __int8 *a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  _BYTE *v6; // r12
  __int64 v7; // r14
  unsigned int v8; // r13d
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r13
  int v11; // r9d
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // r9d
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // r15
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // [rsp+0h] [rbp-E0h]
  int v26; // [rsp+18h] [rbp-C8h]
  _QWORD *v27; // [rsp+18h] [rbp-C8h]
  __int64 v28; // [rsp+18h] [rbp-C8h]
  _BYTE *v29; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+28h] [rbp-B8h]
  _BYTE v31[176]; // [rsp+30h] [rbp-B0h] BYREF

  v4 = *a3;
  if ( (unsigned int)(v4 - 12) <= 1 )
    return sub_ACADE0(*(_QWORD *)(a1 + 8));
  v6 = a2;
  if ( *(_BYTE *)a1 == 14 )
  {
    if ( (unsigned __int8)sub_AC30F0(a2) )
      return a1;
    v4 = *a3;
  }
  if ( (_BYTE)v4 != 17 )
    return 0;
  v7 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v7 + 8) == 18 )
    return 0;
  v8 = *((_DWORD *)a3 + 8);
  if ( v8 > 0x40 )
  {
    if ( v8 - (unsigned int)sub_C444A0(a3 + 24) > 0x40 )
      return sub_ACADE0(v7);
    v9 = **((_QWORD **)a3 + 3);
  }
  else
  {
    v9 = *((_QWORD *)a3 + 3);
  }
  v10 = *(unsigned int *)(v7 + 32);
  v11 = *(_DWORD *)(v7 + 32);
  if ( v10 <= v9 )
    return sub_ACADE0(v7);
  v29 = v31;
  v30 = 0x1000000000LL;
  if ( v10 > 0x10 )
  {
    a2 = v31;
    sub_C8D5F0(&v29, v31, v10, 8);
    v11 = v10;
  }
  v26 = v11;
  v12 = sub_BD5C60(a1, a2, v4);
  v13 = sub_BCB2D0(v12);
  v14 = v26;
  v15 = v13;
  v16 = (_QWORD *)*((_QWORD *)a3 + 3);
  if ( *((_DWORD *)a3 + 8) > 0x40u )
    v16 = (_QWORD *)*v16;
  v27 = v16;
  v17 = 0;
  if ( v14 )
  {
    do
    {
      while ( v17 == v27 )
      {
        v22 = (unsigned int)v30;
        v23 = (unsigned int)v30 + 1LL;
        if ( v23 > HIDWORD(v30) )
        {
          sub_C8D5F0(&v29, v31, v23, 8);
          v22 = (unsigned int)v30;
        }
        v17 = (_QWORD *)((char *)v17 + 1);
        *(_QWORD *)&v29[8 * v22] = v6;
        LODWORD(v30) = v30 + 1;
        if ( (_QWORD *)v10 == v17 )
          goto LABEL_22;
      }
      v18 = sub_ACD640(v15, v17, 0);
      v19 = sub_AD5840(a1, v18, 0);
      v20 = (unsigned int)v30;
      v21 = (unsigned int)v30 + 1LL;
      if ( v21 > HIDWORD(v30) )
      {
        v25 = v19;
        sub_C8D5F0(&v29, v31, v21, 8);
        v20 = (unsigned int)v30;
        v19 = v25;
      }
      v17 = (_QWORD *)((char *)v17 + 1);
      *(_QWORD *)&v29[8 * v20] = v19;
      LODWORD(v30) = v30 + 1;
    }
    while ( (_QWORD *)v10 != v17 );
  }
LABEL_22:
  v24 = (unsigned int)v30;
  result = sub_AD3730(v29, (unsigned int)v30);
  if ( v29 != v31 )
  {
    v28 = result;
    _libc_free(v29, v24);
    return v28;
  }
  return result;
}
