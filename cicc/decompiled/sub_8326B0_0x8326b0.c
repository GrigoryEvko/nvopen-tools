// Function: sub_8326B0
// Address: 0x8326b0
//
_QWORD *__fastcall sub_8326B0(
        __int64 a1,
        __m128i *a2,
        int a3,
        __m128i *a4,
        __int64 *a5,
        _DWORD *a6,
        __int64 *a7,
        _QWORD *a8,
        _BYTE *a9,
        int *a10)
{
  __int64 v12; // r14
  int v13; // ebx
  __int64 i; // r15
  _QWORD *result; // rax
  __int64 v16; // r13
  char v17; // al
  __int64 *v18; // rax
  __int64 v20; // rdi
  const __m128i *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rcx
  __int64 v24; // rdx
  bool v25; // r9
  __int64 v26; // rax
  __m128i *v27; // rdi
  int v28; // eax
  int v29; // [rsp+0h] [rbp-80h]
  __int64 *v30; // [rsp+8h] [rbp-78h]
  int v32; // [rsp+18h] [rbp-68h]
  int v33; // [rsp+1Ch] [rbp-64h]
  unsigned int v35; // [rsp+28h] [rbp-58h]
  int v37; // [rsp+3Ch] [rbp-44h] BYREF
  __m128i *v38; // [rsp+40h] [rbp-40h] BYREF
  __int64 v39[7]; // [rsp+48h] [rbp-38h] BYREF

  v32 = (int)a2;
  v38 = (__m128i *)sub_724DC0();
  if ( a9 )
    v35 = ((a9[40] >> 5) ^ 1) & 1;
  else
    v35 = a10 == 0;
  if ( a5 )
    *a5 = 0;
  v12 = a1;
  v13 = sub_8D3410(a1);
  if ( v13 )
  {
    v13 = 1;
    v12 = sub_8D40F0(a1);
  }
  for ( i = v12; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
    sub_8AE000(i);
  if ( (unsigned int)sub_8D23B0(v12) )
  {
    if ( v35 && (unsigned int)sub_6E5430() )
    {
      a2 = a4;
      sub_685360(0x91Bu, a4, v12);
    }
    goto LABEL_12;
  }
  if ( (unsigned int)sub_8D2BE0(v12) )
  {
    if ( v35 && (unsigned int)sub_6E5430() )
    {
      a2 = a4;
      sub_685360(0xD57u, a4, v12);
    }
    goto LABEL_12;
  }
  if ( (unsigned int)sub_8D2FB0(v12) )
  {
    if ( v35 && (unsigned int)sub_6E5430() )
    {
      a2 = a4;
      sub_6851C0(0x91Cu, a4);
    }
    goto LABEL_12;
  }
  if ( (unsigned int)sub_8D2310(v12) )
  {
    a2 = (__m128i *)v35;
    if ( v35 && (unsigned int)sub_6E5430() )
    {
      a2 = a4;
      sub_6851C0(0x941u, a4);
    }
LABEL_12:
    if ( !a3 )
    {
      if ( !v13 )
      {
        v33 = 1;
LABEL_15:
        *a6 = 0;
        *a7 = 0;
        result = a8;
        *a8 = 0;
        goto LABEL_16;
      }
LABEL_38:
      v33 = v13;
      goto LABEL_15;
    }
    sub_72C970((__int64)v38);
    if ( !v13 )
    {
      v33 = 1;
      goto LABEL_40;
    }
    goto LABEL_24;
  }
  v33 = sub_8D3D40(v12);
  if ( v33 )
  {
    if ( !a3 )
      goto LABEL_36;
    v18 = (__int64 *)sub_6ECAE0(v12, 0, 0, 1, 1u, a4->m128i_i64, v39);
    a2 = v38;
    sub_70FD90(v18, (__int64)v38);
    if ( v13 )
    {
LABEL_60:
      v13 = 0;
LABEL_24:
      v33 = v13;
      v16 = sub_6EAFA0(1u);
      goto LABEL_25;
    }
LABEL_58:
    v33 = 0;
    goto LABEL_40;
  }
  if ( !(unsigned int)sub_8D3A70(v12) )
  {
    while ( 1 )
    {
      v17 = *(_BYTE *)(v12 + 140);
      if ( v17 != 12 )
        break;
      v12 = *(_QWORD *)(v12 + 160);
    }
    if ( !v17 )
      goto LABEL_12;
    if ( !a3 )
    {
LABEL_36:
      if ( !v13 )
      {
        v33 = 0;
        goto LABEL_15;
      }
      v13 = 0;
      goto LABEL_38;
    }
    a2 = v38;
    if ( (unsigned int)sub_72FDF0(i, v38) )
      goto LABEL_57;
LABEL_100:
    sub_721090();
  }
  if ( !(unsigned int)sub_8D3AD0(v12) )
  {
    if ( !a3 )
      goto LABEL_36;
    v20 = 0;
    goto LABEL_64;
  }
  v23 = 0;
  if ( !v35 )
    v23 = v39;
  LODWORD(v24) = i;
  if ( a9 && (a9[43] & 8) != 0 )
    v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 40LL) + 32LL);
  v25 = 0;
  if ( a3 )
  {
    v29 = v24;
    v30 = v23;
    v28 = sub_6E6010();
    LODWORD(v24) = v29;
    v23 = v30;
    v25 = v28 != 0;
  }
  a2 = a4;
  v26 = sub_87CAB0(
          i,
          (_DWORD)a4,
          v24,
          0,
          (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0,
          v25,
          v32,
          (__int64)v23,
          (__int64)&v37);
  v20 = v26;
  if ( v35 )
  {
    if ( v37 )
      goto LABEL_12;
  }
  else if ( LODWORD(v39[0]) )
  {
    goto LABEL_12;
  }
  if ( a5 )
    *a5 = v26;
  if ( v26 )
  {
    if ( !a3 )
      goto LABEL_36;
LABEL_64:
    a2 = 0;
    v16 = sub_6F5430(v20, 0, v12, 0, 1, 0, 1u, 0, 1u, 0, (__int64)a4);
    v21 = (const __m128i *)sub_730250(v16);
    if ( !v21 || (v21[10].m128i_i8[11] & 2) == 0 )
      goto LABEL_66;
    a2 = v38;
    sub_72A510(v21, v38);
    if ( a9 && *(char *)(v16 + 50) < 0 )
    {
      a9[41] |= 0x10u;
      if ( v13 )
        goto LABEL_60;
      goto LABEL_58;
    }
LABEL_57:
    if ( v13 )
      goto LABEL_60;
    goto LABEL_58;
  }
  if ( !a3 )
    goto LABEL_36;
  v16 = sub_6EAFA0(1u);
  if ( word_4D04898 && *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
  {
    a2 = v38;
    if ( (unsigned int)sub_72FDF0(i, v38) )
    {
      v27 = v38;
      v38[10].m128i_i8[11] |= 2u;
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
      {
        a2 = (__m128i *)v16;
        sub_71AAB0((__int64)v27, v16);
        if ( v13 )
          goto LABEL_60;
        goto LABEL_58;
      }
      goto LABEL_57;
    }
    goto LABEL_100;
  }
LABEL_66:
  if ( v13 )
  {
    if ( !v16 )
      goto LABEL_60;
    if ( *(_BYTE *)(v16 + 48) == 1 )
      goto LABEL_26;
    if ( dword_4D048B8 && (unsigned int)sub_8D3A70(i) )
    {
      if ( v35 )
      {
        v22 = sub_6EB250(i, i, (int)a4, 0, 0);
      }
      else
      {
        v22 = sub_6EB250(i, i, (int)a4, 0, (__int64)v39);
        v33 = LODWORD(v39[0]) != 0;
      }
    }
    else
    {
      v22 = 0;
    }
    a2 = (__m128i *)a1;
    v16 = sub_6EB060(v16, a1, v22);
  }
  else
  {
    v33 = 0;
  }
LABEL_25:
  if ( v16 )
  {
LABEL_26:
    *a6 = 0;
    result = a7;
    *a7 = v16;
    goto LABEL_16;
  }
LABEL_40:
  *a6 = 1;
  result = (_QWORD *)sub_724E50((__int64 *)&v38, a2);
  *a8 = result;
LABEL_16:
  if ( a10 )
  {
    result = a10;
    *a10 = v33;
  }
  if ( v38 )
    return sub_724E30((__int64)&v38);
  return result;
}
