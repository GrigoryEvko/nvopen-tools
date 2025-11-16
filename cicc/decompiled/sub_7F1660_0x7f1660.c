// Function: sub_7F1660
// Address: 0x7f1660
//
__m128i **__fastcall sub_7F1660(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __m128i **a4,
        __int64 **a5,
        int *a6,
        __m128i **a7,
        _QWORD *a8,
        unsigned int a9)
{
  _QWORD *v10; // r8
  _DWORD *v11; // r9
  const __m128i *v12; // r14
  __int64 v13; // r15
  __int64 v14; // rbx
  __int64 i; // r13
  __int8 v16; // al
  int v17; // ebx
  __int64 j; // r14
  __m128i *v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 *v25; // rdi
  __m128i *v26; // rdi
  __m128i *v28; // rax
  const __m128i *v29; // rdi
  int v30; // eax
  __int64 v31; // rsi
  __m128i *v32; // rax
  int v33; // ecx
  __m128i *v34; // rax
  int v38; // [rsp+18h] [rbp-48h]
  char v39; // [rsp+1Fh] [rbp-41h]
  __m128i *v40; // [rsp+28h] [rbp-38h] BYREF

  v38 = a2;
  sub_7E0590(a1[9]);
  *v10 = 0;
  *v11 = 0;
  v12 = (const __m128i *)a1[9];
  v39 = *((_BYTE *)a1 + 56);
  if ( (*((_BYTE *)a1 + 25) & 1) != 0 || !(unsigned int)sub_8D32B0(*a1) )
  {
    v13 = v12->m128i_i64[0];
    v14 = *a1;
  }
  else
  {
    v13 = sub_8D46C0(v12->m128i_i64[0]);
    v14 = sub_8D46C0(*a1);
  }
  while ( *(_BYTE *)(v13 + 140) == 12 )
    v13 = *(_QWORD *)(v13 + 160);
  for ( ; *(_BYTE *)(v14 + 140) == 12; v14 = *(_QWORD *)(v14 + 160) )
    ;
  sub_7E3EE0(v13);
  sub_7E3EE0(v14);
  if ( v39 == 15 )
  {
    for ( i = **(_QWORD **)(v14 + 168); (*(_BYTE *)(i + 96) & 3) == 0 || v13 != *(_QWORD *)(i + 40); i = *(_QWORD *)i )
      ;
    v16 = v12[1].m128i_i8[8];
    if ( a3 )
    {
LABEL_13:
      if ( v16 != 1 || v39 != v12[3].m128i_i8[8] )
      {
        v17 = 0;
        v40 = (__m128i *)sub_7EC2A0(v12, a2);
        sub_7EE560(v40, (__m128i *)a9);
        *a8 = 0;
LABEL_16:
        for ( j = **(_QWORD **)(v13 + 168); *(_QWORD *)(j + 40) != a3 || (*(_BYTE *)(j + 96) & 2) == 0; j = *(_QWORD *)j )
          ;
        v19 = v40;
        v20 = 0;
        *a5 = (__int64 *)j;
        if ( v13 == sub_8D7160(v19, 0) )
        {
          *a6 = 1;
          v30 = 1;
          if ( !*(_QWORD *)(j + 104) )
            v30 = v38;
          v38 = v30;
        }
LABEL_21:
        v24 = a9;
        v25 = (__int64 *)v40;
        if ( a9 )
        {
LABEL_22:
          *a4 = 0;
          goto LABEL_23;
        }
        goto LABEL_39;
      }
      goto LABEL_55;
    }
  }
  else
  {
    for ( i = **(_QWORD **)(v13 + 168); (*(_BYTE *)(i + 96) & 3) == 0 || *(_QWORD *)(i + 40) != v14; i = *(_QWORD *)i )
      ;
    v16 = v12[1].m128i_i8[8];
    if ( a3 )
      goto LABEL_13;
  }
  if ( (*(_BYTE *)(i + 96) & 2) == 0 )
  {
    v33 = 1;
    if ( !*(_QWORD *)(i + 104) )
      v33 = a2;
    v38 = v33;
    if ( v16 != 1 || v39 != v12[3].m128i_i8[8] )
    {
      v17 = 0;
      v34 = (__m128i *)sub_7EC2A0(v12, a2);
      v20 = a9;
      v40 = v34;
      sub_7EE560(v34, (__m128i *)a9);
      *a8 = 0;
      goto LABEL_21;
    }
LABEL_55:
    v17 = 0;
LABEL_56:
    sub_7F1660((_DWORD)v12, v38, a3, (_DWORD)a4, (_DWORD)a5, (_DWORD)a6, (__int64)&v40, (__int64)a8, a9);
    v25 = (__int64 *)v40;
    goto LABEL_23;
  }
  a3 = *(_QWORD *)(i + 40);
  if ( v16 == 1 )
  {
    v17 = 1;
    if ( v12[3].m128i_i8[8] == v39 )
      goto LABEL_56;
  }
  v17 = 1;
  v28 = (__m128i *)sub_7EC2A0(v12, a2);
  v20 = a9;
  v40 = v28;
  sub_7EE560(v28, (__m128i *)a9);
  *a8 = 0;
  if ( a3 )
    goto LABEL_16;
  v24 = a9;
  v25 = (__int64 *)v40;
  if ( a9 )
    goto LABEL_22;
LABEL_39:
  if ( (unsigned int)sub_7311F0((__int64)v25, v20, v21, v24, v22, v23) )
  {
    v25 = (__int64 *)v40;
    *a4 = 0;
  }
  else if ( a3 && !*a6 || v38 )
  {
    v29 = v40;
    *a4 = v40;
    v40 = (__m128i *)sub_7E8090(v29, 0);
    v25 = (__int64 *)v40;
  }
  else
  {
    v25 = (__int64 *)v40;
    *a4 = 0;
  }
LABEL_23:
  sub_8D46C0(*v25);
  if ( v39 == 15 )
  {
    v26 = v40;
    *a8 += *(_QWORD *)(i + 104);
  }
  else
  {
    v26 = v40;
    if ( *a5 )
    {
      if ( v17 )
      {
        v40 = (__m128i *)sub_73DCD0(v40);
        v31 = (__int64)*a5;
        v40 = (__m128i *)sub_7E85E0(v40, *a5, *a6);
        v32 = (__m128i *)sub_73E1B0((__int64)v40, v31);
        *a5 = 0;
        v26 = v32;
      }
    }
    else
    {
      v40 = (__m128i *)sub_73DCD0(v40);
      v40 = sub_7E8750(v40, i, 0);
      v26 = (__m128i *)sub_73E1B0((__int64)v40, i);
    }
  }
  *a7 = v26;
  return a7;
}
