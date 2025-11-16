// Function: sub_25BF070
// Address: 0x25bf070
//
unsigned __int64 __fastcall sub_25BF070(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 i; // r14
  char v9; // r12
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax
  int v19; // eax
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rdi
  int v24; // ecx
  int v25; // ecx
  unsigned int v26; // edx
  __int64 v27; // rsi
  int v28; // r12d
  unsigned __int8 v30; // [rsp+13h] [rbp-8Dh]
  unsigned int v31; // [rsp+14h] [rbp-8Ch]
  unsigned int v33; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v34; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h] BYREF
  __m128i v36[3]; // [rsp+30h] [rbp-70h] BYREF
  char v37; // [rsp+60h] [rbp-40h]

  v31 = sub_CF5E30(a3, a1);
  if ( !v31 )
    return 0;
  if ( !a2 )
    return v31;
  v4 = *(_QWORD *)(a1 + 120);
  v33 = 0;
  v34 = 0;
  v35 = v4;
  if ( (unsigned __int8)sub_A74390(&v35, 83, 0)
    || (v36[0].m128i_i64[0] = *(_QWORD *)(a1 + 120), (unsigned __int8)sub_A74390(v36[0].m128i_i64, 84, 0)) )
  {
    v33 |= 3u;
  }
  v5 = *(_QWORD *)(a1 + 80);
  v6 = a1 + 72;
  if ( a1 + 72 == v5 )
  {
    i = 0;
    while ( 1 )
    {
LABEL_13:
      if ( v6 == v5 )
        return v33 & v31 | ((unsigned __int64)v34 << 32);
      if ( !i )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(i - 24) - 34) <= 0x33u
        && (v12 = 0x8000000000041LL, _bittest64(&v12, (unsigned int)*(unsigned __int8 *)(i - 24) - 34)) )
      {
        if ( *(char *)(i - 17) < 0 )
        {
          v13 = sub_BD2BC0(i - 24);
          v15 = v13 + v14;
          v16 = 0;
          if ( *(char *)(i - 17) < 0 )
            v16 = sub_BD2BC0(i - 24);
          if ( (unsigned int)((v15 - v16) >> 4) )
            goto LABEL_38;
        }
        v17 = *(_QWORD *)(i - 56);
        if ( !v17 || *(_BYTE *)v17 || *(_QWORD *)(v17 + 24) != *(_QWORD *)(i + 56) )
          goto LABEL_38;
        v36[0].m128i_i64[0] = *(_QWORD *)(i - 56);
        if ( *(_DWORD *)(a4 + 16) )
        {
          v23 = *(_QWORD *)(a4 + 8);
          v24 = *(_DWORD *)(a4 + 24);
          if ( !v24 )
            goto LABEL_38;
          v25 = v24 - 1;
          v26 = v25 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v27 = *(_QWORD *)(v23 + 8LL * v26);
          if ( v17 != v27 )
          {
            v28 = 1;
            while ( v27 != -4096 )
            {
              v26 = v25 & (v28 + v26);
              v27 = *(_QWORD *)(v23 + 8LL * v26);
              if ( v17 == v27 )
                goto LABEL_47;
              ++v28;
            }
LABEL_38:
            v18 = sub_CF5CA0(a3, i - 24);
            if ( v18 )
            {
              if ( *(_BYTE *)(i - 24) != 85
                || (v22 = *(_QWORD *)(i - 56)) == 0
                || *(_BYTE *)v22
                || *(_QWORD *)(v22 + 24) != *(_QWORD *)(i + 56)
                || (*(_BYTE *)(v22 + 33) & 0x20) == 0
                || *(_DWORD *)(v22 + 36) != 291 )
              {
                v33 |= ((unsigned __int8)v18 >> 6) | v18 & 0xFFFFFFFC;
                v19 = v18 & 3;
                if ( v19 )
                  sub_25BD270((int *)&v33, (unsigned __int8 *)(i - 24), v19, a3);
              }
            }
            goto LABEL_22;
          }
        }
        else
        {
          v20 = *(_QWORD **)(a4 + 32);
          v21 = &v20[*(unsigned int *)(a4 + 40)];
          if ( v21 == sub_25BD100(v20, (__int64)v21, v36[0].m128i_i64) )
            goto LABEL_38;
        }
LABEL_47:
        sub_25BD270((int *)&v34, (unsigned __int8 *)(i - 24), 3, a3);
      }
      else
      {
        v9 = sub_B46490(i - 24);
        v10 = sub_B46420(i - 24);
        if ( v9 )
        {
          v30 = 2 - ((v10 == 0) - 1);
          goto LABEL_18;
        }
        if ( v10 )
        {
          v30 = 1;
LABEL_18:
          sub_D66840(v36, (_BYTE *)(i - 24));
          if ( v37 )
          {
            if ( sub_B46560((unsigned __int8 *)(i - 24)) )
              v33 |= 4 * v30;
            sub_25BCA40((int *)&v33, (unsigned __int8 **)v36, v30, a3);
          }
          else
          {
            v33 |= (16 * v30) | v30 | (4 * v30) | (v30 << 6);
          }
        }
      }
LABEL_22:
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v5 + 32) )
      {
        v11 = v5 - 24;
        if ( !v5 )
          v11 = 0;
        if ( i != v11 + 48 )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v6 == v5 )
          return v33 & v31 | ((unsigned __int64)v34 << 32);
        if ( !v5 )
          BUG();
      }
    }
  }
  if ( !v5 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v5 + 32);
    if ( i != v5 + 24 )
      goto LABEL_13;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v6 == v5 )
      return v33 & v31 | ((unsigned __int64)v34 << 32);
    if ( !v5 )
      BUG();
  }
}
