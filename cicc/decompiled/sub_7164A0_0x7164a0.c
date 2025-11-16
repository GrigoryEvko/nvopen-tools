// Function: sub_7164A0
// Address: 0x7164a0
//
unsigned __int64 __fastcall sub_7164A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int *a5)
{
  __int64 v5; // r9
  unsigned int v6; // r14d
  __m128i *v7; // r13
  int *v9; // rbx
  __int64 v10; // rdi
  unsigned __int64 result; // rax
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // r15
  __int64 v15; // rsi
  unsigned int v16; // r9d
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 i; // r8
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 j; // rsi
  int v25; // eax
  int v26; // eax
  int v27; // eax
  const __m128i *v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdi
  unsigned int v32; // [rsp+4h] [rbp-5Ch]
  unsigned int v33; // [rsp+8h] [rbp-58h]
  unsigned int v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  char v36; // [rsp+1Bh] [rbp-45h] BYREF
  int v37; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v38; // [rsp+20h] [rbp-40h] BYREF
  __int64 v39[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = (unsigned int)a3;
  v6 = a4;
  v7 = (__m128i *)a2;
  v9 = a5;
  if ( !a5 )
    v9 = &v37;
  *v9 = 0;
  v10 = dword_4D03F94;
  if ( dword_4D03F94 || (v10 = (__int64)a1, v33 = a3, v12 = sub_716120((__int64)a1, a2), v5 = v33, !v12) )
  {
    switch ( *((_BYTE *)a1 + 24) )
    {
      case 0:
        v10 = a2;
        sub_72C970(a2);
        if ( v9 == &v37 )
          goto LABEL_15;
        goto LABEL_11;
      case 1:
        v34 = v5;
        v14 = a1[9];
        v39[0] = sub_724DC0(v10, a2, a3, a4, a5, v5);
        v15 = v39[0];
        v16 = v34;
        switch ( *((_BYTE *)a1 + 56) )
        {
          case 0:
            a2 = (__int64)v7;
            if ( (unsigned int)sub_716BB0(v14, v7, v34, v6, v9) )
              goto LABEL_44;
            goto LABEL_6;
          case 4:
            if ( *(_BYTE *)(v14 + 24) != 2 )
              goto LABEL_6;
            v31 = *(_QWORD *)(v14 + 56);
            if ( *(_BYTE *)(v31 + 173) != 6 || *(_BYTE *)(v31 + 176) )
              goto LABEL_6;
            a2 = (__int64)v7;
            sub_72A510(v31, v7);
            goto LABEL_44;
          case 5:
            if ( !(unsigned int)sub_8D2E30(*a1) || !(unsigned int)sub_8D2E30(*(_QWORD *)v14) )
              goto LABEL_6;
            for ( i = sub_8D46C0(*a1); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            v32 = v34;
            v35 = i;
            v22 = sub_8D46C0(*(_QWORD *)v14);
            v16 = v32;
            for ( j = v22; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            if ( v35 != j )
            {
              v25 = sub_8D97D0(v35, j, 0, v23, v35);
              v16 = v32;
              if ( !v25 )
              {
                v26 = sub_8D29E0(v35);
                v16 = v32;
                if ( !v26 )
                {
                  if ( !dword_4F077C0 )
                    goto LABEL_6;
                  v27 = sub_8D2600(v35);
                  v16 = v32;
                  if ( !v27 )
                    goto LABEL_6;
                }
              }
            }
            v15 = v39[0];
LABEL_58:
            if ( !(unsigned int)sub_7164A0(v14, v15, v16, v6, v9) || *v9 )
            {
LABEL_6:
              sub_724E30(v39);
              return 0;
            }
            sub_724C70(v7, 0);
            v28 = (const __m128i *)v39[0];
            a2 = (__int64)v7;
            v7[8].m128i_i64[0] = *a1;
            sub_710F60(
              v28,
              v7,
              0,
              0,
              (*((_BYTE *)a1 + 27) & 2) != 0,
              1,
              (*((_BYTE *)a1 + 58) & 2) != 0,
              (v6 >> 1) & 1,
              (_DWORD *)&v38 + 1,
              dword_4F07508,
              (unsigned int *)&v38,
              &v36);
            if ( v38 )
            {
              sub_724E30(v39);
              return 0;
            }
LABEL_44:
            v10 = (__int64)v39;
            sub_724E30(v39);
            if ( v9 != &v37 )
              goto LABEL_11;
            break;
          case 0xE:
            goto LABEL_58;
          case 0x15:
            if ( (*(_BYTE *)(v14 + 25) & 3) == 0 )
              goto LABEL_6;
            a2 = (__int64)v7;
            if ( !(unsigned int)sub_716BB0(v14, v7, v34, v6, v9) )
              goto LABEL_6;
            if ( !(unsigned int)sub_8D2E30(v7[8].m128i_i64[0]) )
              goto LABEL_6;
            v29 = sub_8D46C0(v7[8].m128i_i64[0]);
            if ( !(unsigned int)sub_8D3410(v29) )
              goto LABEL_6;
            v30 = sub_8D67C0(v29);
            v7[10].m128i_i8[8] |= 8u;
            v7[8].m128i_i64[0] = v30;
            goto LABEL_44;
          case 0x32:
          case 0x33:
            a2 = (__int64)v7;
            if ( (unsigned int)sub_716A40(a1, v7, v34, v6, v9) )
              goto LABEL_44;
            goto LABEL_6;
          default:
            goto LABEL_6;
        }
        goto LABEL_15;
      case 2:
        v10 = a1[7];
        goto LABEL_20;
      case 3:
        if ( !word_4D04898 )
          return 0;
        v10 = sub_6EA7C0(a1[7]);
        if ( !v10 )
          return 0;
LABEL_20:
        sub_72A510(v10, a2);
        if ( v9 == &v37 )
          goto LABEL_15;
        goto LABEL_11;
      case 0x14:
        v10 = a1[7];
        if ( (*(_BYTE *)(v10 + 193) & 4) != 0 )
          return 0;
        sub_70D050(v10, a2, (unsigned int)v5, v9);
        if ( v9 == &v37 )
          goto LABEL_15;
        goto LABEL_11;
      case 0x18:
        if ( !word_4D04898 )
          return 0;
        if ( (v6 & 4) != 0 )
          return 0;
        if ( !qword_4F078B8 )
          return 0;
        if ( *((_DWORD *)a1 + 14) )
          return 0;
        v17 = a1[2];
        if ( !v17 )
          return 0;
        if ( *(_BYTE *)(v17 + 24) != 4 )
          return 0;
        v18 = *(_QWORD *)(v17 + 56);
        if ( (*(_BYTE *)(v18 + 89) & 4) == 0 )
          return 0;
        v19 = *(_QWORD *)(*(_QWORD *)(v18 + 40) + 32LL);
        if ( !v19 )
          return 0;
        break;
      default:
        return 0;
    }
    while ( 1 )
    {
      v20 = (_QWORD *)qword_4F078B8;
      if ( qword_4F078B8 )
        break;
LABEL_71:
      if ( (*(_BYTE *)(v19 + 89) & 4) != 0 )
      {
        v19 = *(_QWORD *)(*(_QWORD *)(v19 + 40) + 32LL);
        if ( v19 )
          continue;
      }
      return 0;
    }
    while ( 1 )
    {
      v10 = v20[1];
      if ( v10 )
      {
        if ( *(_QWORD *)(v10 + 128) == v19 )
          break;
      }
      v20 = (_QWORD *)*v20;
      if ( !v20 )
        goto LABEL_71;
    }
    sub_72D410(v10, a2);
    if ( v9 != &v37 )
      goto LABEL_11;
  }
  else if ( v9 != &v37 )
  {
LABEL_11:
    LOBYTE(v13) = v7[10].m128i_i8[13];
    goto LABEL_12;
  }
LABEL_15:
  v13 = v7[10].m128i_u8[13];
  if ( v37 )
  {
    if ( (_BYTE)v13 == 12 )
      return (0x1042uLL >> v13) & 1;
    v39[0] = sub_724DC0(v10, a2, a3, v13, a5, v5);
    sub_72A510(v7, v39[0]);
    sub_70FDD0(v39[0], (__int64)v7, v7[8].m128i_i64[0], 0);
    sub_724E30(v39);
    LOBYTE(v13) = v7[10].m128i_i8[13];
  }
LABEL_12:
  result = 0;
  if ( (unsigned __int8)v13 <= 0xCu )
    return (0x1042uLL >> v13) & 1;
  return result;
}
