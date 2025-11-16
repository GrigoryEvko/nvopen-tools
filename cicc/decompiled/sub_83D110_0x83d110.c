// Function: sub_83D110
// Address: 0x83d110
//
__int64 __fastcall sub_83D110(
        const __m128i *a1,
        __m128i *a2,
        int a3,
        __m128i *a4,
        __int64 a5,
        int a6,
        __int64 *a7,
        __int64 *a8,
        _DWORD *a9)
{
  int v10; // r13d
  __m128i *v12; // rbx
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 *v16; // rdi
  __int64 v17; // rax
  char k; // dl
  __int64 v19; // r14
  __int64 v20; // rax
  bool v21; // zf
  unsigned int v22; // r12d
  __int64 v23; // rax
  __int64 v25; // rax
  __int8 i; // al
  __int64 v27; // rax
  __m128i *v28; // rax
  __int8 v29; // dl
  const __m128i *v30; // rax
  __int8 j; // dl
  __int64 v32; // rax
  const __m128i *v33; // rbx
  _QWORD *v34; // r14
  __int64 v35; // rax
  __int64 v36; // rdi
  _DWORD *v37; // r12
  __int64 v40; // [rsp+18h] [rbp-C8h]
  int v41; // [rsp+24h] [rbp-BCh] BYREF
  __int64 *v42; // [rsp+28h] [rbp-B8h] BYREF
  const __m128i *v43; // [rsp+30h] [rbp-B0h] BYREF
  __m128i *v44; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v45; // [rsp+40h] [rbp-A0h] BYREF
  __m128i *v46; // [rsp+48h] [rbp-98h] BYREF
  int v47[36]; // [rsp+50h] [rbp-90h] BYREF

  v10 = (int)a1;
  v12 = a2;
  *a7 = 0;
  v42 = 0;
  *a8 = 0;
  v43 = a1;
  v45 = 0;
  v46 = 0;
  v41 = 0;
  *a9 = 0;
  if ( !a2 )
  {
    v12 = (__m128i *)sub_8D4940(a1);
    for ( i = v12[8].m128i_i8[12]; i == 12; i = v12[8].m128i_i8[12] )
      v12 = (__m128i *)v12[10].m128i_i64[0];
    if ( !i )
    {
      v27 = sub_72C930();
      v22 = 1;
      v16 = v42;
      *a8 = v27;
      *a7 = v27;
      goto LABEL_34;
    }
  }
  if ( a5 )
  {
    if ( !*(_BYTE *)(a5 + 8) )
    {
      a4 = (__m128i *)(*(_QWORD *)(a5 + 24) + 8LL);
      goto LABEL_5;
    }
  }
  else
  {
    if ( a4[1].m128i_i8[0] != 5 )
    {
LABEL_5:
      if ( (a4[1].m128i_i8[3] & 8) != 0 && a4[1].m128i_i8[0] == 3 )
        sub_6F6890(a4, 0);
      v44 = (__m128i *)a4->m128i_i64[0];
      if ( (unsigned int)sub_8D2690(v44) )
      {
        a4 = 0;
        v44 = (__m128i *)sub_72C570();
      }
      if ( dword_4F04C44 == -1 )
      {
        v25 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v25 + 6) & 6) == 0 && *(_BYTE *)(v25 + 4) != 12 )
        {
          v15 = sub_880AD0(v12->m128i_i64[0]);
          goto LABEL_42;
        }
      }
      a5 = 0;
      goto LABEL_10;
    }
    a5 = a4[9].m128i_i64[0];
  }
  v44 = 0;
  if ( dword_4F04C44 != -1 || (v23 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v23 + 6) & 6) != 0) )
  {
    a4 = 0;
LABEL_32:
    if ( (unsigned int)sub_82ED80(a5) )
      goto LABEL_33;
    goto LABEL_12;
  }
  a4 = 0;
  if ( *(_BYTE *)(v23 + 4) != 12 )
  {
    v14 = sub_880AD0(v12->m128i_i64[0]);
    v15 = v14;
    if ( !a5 )
      goto LABEL_42;
    goto LABEL_14;
  }
LABEL_10:
  if ( !v44 )
    goto LABEL_32;
  if ( (unsigned int)sub_8DBE70(v44) )
  {
LABEL_33:
    *a9 = 1;
    v16 = v42;
    v22 = 0;
    goto LABEL_34;
  }
LABEL_12:
  v12[8].m128i_i8[12] = 21;
  v13 = sub_8DBE70(v43);
  v12[8].m128i_i8[12] = 14;
  if ( v13 )
    goto LABEL_33;
  v14 = sub_880AD0(v12->m128i_i64[0]);
  v15 = v14;
  if ( !a5 )
  {
LABEL_42:
    if ( (unsigned int)sub_82C250(&v43, &v44, (__int64)a4, v15, 0, &v45, &v46, 0)
      && (unsigned int)sub_828690((__int64)v43, (__int64)v44, v45, (__int64)v46, (__int64)&v42, v15) )
    {
      v16 = v42;
      if ( v42 )
      {
        v19 = v42[4];
LABEL_22:
        *a8 = v19;
        if ( a3 )
        {
          v33 = *(const __m128i **)(v12[10].m128i_i64[1] + 32);
          v34 = sub_7259C0(12);
          v35 = *a8;
          *((_BYTE *)v34 + 184) = 3;
          v34[20] = v35;
          if ( v33 )
          {
            sub_7296C0(v47);
            v40 = v34[21];
            *(_QWORD *)(v40 + 24) = sub_73B8B0(v33, 0);
            sub_729730(v47[0]);
          }
          v42[4] = (__int64)v34;
        }
        sub_892150(v47);
        v20 = sub_8A2270(v10, (_DWORD)v42, v15, a6, 0, (unsigned int)&v41, (__int64)v47);
        v16 = v42;
        v21 = v41 == 0;
        *a7 = v20;
        v22 = v21;
        goto LABEL_34;
      }
      goto LABEL_49;
    }
LABEL_43:
    v16 = v42;
    goto LABEL_44;
  }
LABEL_14:
  if ( !(unsigned int)sub_83CDB0(a5, v12, v14, (__int64 *)&v42) )
    goto LABEL_43;
  v16 = v42;
  if ( !v42 )
  {
LABEL_49:
    v28 = v44;
    if ( v44 )
    {
      while ( 1 )
      {
        v29 = v28[8].m128i_i8[12];
        if ( v29 != 12 )
          break;
        v28 = (__m128i *)v28[10].m128i_i64[0];
      }
      if ( !v29 )
        goto LABEL_57;
    }
    v30 = v43;
    for ( j = v43[8].m128i_i8[12]; j == 12; j = v30[8].m128i_i8[12] )
      v30 = (const __m128i *)v30[10].m128i_i64[0];
    if ( !j )
    {
LABEL_57:
      v32 = sub_72C930();
      v16 = v42;
      v22 = 1;
      *a8 = v32;
      *a7 = v32;
      goto LABEL_34;
    }
LABEL_44:
    v22 = 0;
    goto LABEL_34;
  }
  v17 = v42[4];
  for ( k = *(_BYTE *)(v17 + 140); k == 12; k = *(_BYTE *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  if ( !k )
  {
    v36 = (__int64)v42;
    *(_QWORD *)(v36 + 32) = sub_72C930();
    v19 = v42[4];
    goto LABEL_22;
  }
  if ( qword_4F07320[0] )
  {
    if ( *(_QWORD *)&dword_4D04988 )
      goto LABEL_21;
    sub_8865A0("initializer_list");
  }
  if ( *(_QWORD *)&dword_4D04988 )
  {
LABEL_21:
    *(_QWORD *)v47 = sub_725090(0);
    *(_QWORD *)(*(_QWORD *)v47 + 32LL) = v42[4];
    v19 = *(_QWORD *)(sub_8A0370(dword_4D04988, (unsigned int)v47, 0, 0, 0, 0, 0) + 88);
    v42[4] = v19;
    if ( dword_4F077C4 == 2 )
    {
      if ( (unsigned int)sub_8D23B0(v19) && (unsigned int)sub_8D3A70(v19) )
        sub_8AD220(v19, 0);
      v19 = v42[4];
    }
    goto LABEL_22;
  }
  v37 = (_DWORD *)sub_6E1A20(a5);
  if ( (unsigned int)sub_6E5430() )
    sub_6851C0(0x930u, v37);
  v16 = v42;
  v22 = 0;
LABEL_34:
  if ( v16 )
    sub_725130(v16);
  return v22;
}
