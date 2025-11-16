// Function: sub_7E2820
// Address: 0x7e2820
//
void __fastcall sub_7E2820(const __m128i *a1, __int64 a2, __int64 a3, _DWORD *a4, int *a5)
{
  __int64 j; // r9
  __int64 v9; // r9
  char v10; // dl
  unsigned __int64 v11; // r12
  __int64 i; // rax
  __int64 v13; // rsi
  __int64 v14; // r10
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  bool v17; // cl
  unsigned __int64 v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // r12
  __int64 v21; // rax
  _BYTE *v22; // r15
  _QWORD *v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  _QWORD *v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  char v31; // [rsp+10h] [rbp-70h]
  int v33; // [rsp+24h] [rbp-5Ch] BYREF
  __int64 v34; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v35[80]; // [rsp+30h] [rbp-50h] BYREF

  if ( *(_BYTE *)(a2 + 140) == 10 )
  {
    v9 = sub_72FD90(*(_QWORD *)(a2 + 160), 11);
    if ( !v9 )
    {
      v17 = 0;
      v10 = 0;
      v11 = 0;
LABEL_22:
      v18 = *(_QWORD *)(a2 + 128);
      if ( v18 > v11 || v17 )
      {
        if ( !a4 )
        {
          sub_7E26B0(a1, v11 + a3, v18 - v11, v10, a5);
          return;
        }
        goto LABEL_6;
      }
      return;
    }
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      for ( i = *(_QWORD *)(v9 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v13 = *(_QWORD *)(v9 + 128);
      if ( v13 == v11 )
      {
        if ( *(_BYTE *)(v9 + 136) == v10 )
          goto LABEL_15;
        if ( a4 )
        {
LABEL_29:
          *a4 = 1;
          if ( *(_QWORD *)(a2 + 128) <= v11 && !v10 )
            return;
LABEL_6:
          *a4 = 1;
          return;
        }
      }
      else if ( a4 )
      {
        goto LABEL_29;
      }
      v26 = i;
      v29 = v9;
      sub_7E26B0(a1, a3 + v11, v13 - v11, v10, a5);
      i = v26;
      v9 = v29;
LABEL_15:
      if ( (*(_BYTE *)(i + 140) & 0xFD) == 8 )
      {
        v27 = v9;
        v30 = i;
        sub_7E2820(a1, i, a3 + v11, a4, a5);
        v9 = v27;
        i = v30;
      }
      v14 = *(_QWORD *)(v9 + 128);
      v15 = *(_BYTE *)(v9 + 144) & 4;
      if ( v15 )
      {
        v15 = *(_BYTE *)(v9 + 136) + *(unsigned __int8 *)(v9 + 137) % dword_4F06BA0;
        v11 = *(unsigned __int8 *)(v9 + 137) / dword_4F06BA0 + v14;
        if ( dword_4F06BA0 == v15 )
        {
          ++v11;
          v15 = 0;
        }
      }
      else
      {
        v11 = *(_QWORD *)(i + 128) + v14;
      }
      v31 = v15;
      v16 = sub_72FD90(*(_QWORD *)(v9 + 112), 11);
      v10 = v31;
      v9 = v16;
      if ( !v16 )
      {
        v17 = v31 != 0;
        goto LABEL_22;
      }
    }
  }
  v33 = 0;
  for ( j = sub_8D4050(a2); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v28 = (_QWORD *)j;
  sub_7E2820(a1, j, a3, &v33, a5);
  if ( v33 )
  {
    if ( a4 )
      goto LABEL_6;
    v19 = sub_73B8B0(a1, 0);
    v20 = sub_7FC780(v28, &v34, v35);
    v21 = sub_72D2E0(v28);
    v22 = sub_73E110((__int64)v19, v21);
    if ( (unsigned int)sub_8D4070(a2) )
      v23 = sub_7D78E0(a2);
    else
      v23 = sub_73A830(*(_QWORD *)(a2 + 176), byte_4F06A51[0]);
    *((_QWORD *)v22 + 2) = v23;
    v24 = sub_7F88E0(v20, v22);
    sub_7E25D0(v24, a5);
    v25 = sub_73E830(v34);
    sub_7E2820(v25, v28, a3, 0, v35);
  }
}
