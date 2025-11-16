// Function: sub_716A40
// Address: 0x716a40
//
__int64 __fastcall sub_716A40(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, int *a5, __int64 a6)
{
  unsigned int v6; // r14d
  _QWORD *v8; // r15
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned int v15; // r12d
  const __m128i *v17; // r10
  __int8 v18; // al
  __int8 v19; // dl
  const __m128i *v20; // [rsp+8h] [rbp-68h]
  unsigned int v21; // [rsp+1Ch] [rbp-54h]
  char v22; // [rsp+27h] [rbp-49h] BYREF
  int v23; // [rsp+28h] [rbp-48h] BYREF
  int v24; // [rsp+2Ch] [rbp-44h] BYREF
  const __m128i *v25; // [rsp+30h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a4;
  v8 = *(_QWORD **)(a1 + 72);
  v21 = a3;
  v9 = v8[2];
  v25 = (const __m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  v14 = sub_724DC0(a1, a2, v10, v11, v12, v13);
  *a5 = 0;
  v26[0] = v14;
  if ( (*(_BYTE *)(a1 + 59) & 0x20) != 0 )
  {
    v9 = *(_QWORD *)(a1 + 72);
    v8 = *(_QWORD **)(v9 + 16);
  }
  if ( word_4D04898 )
  {
    if ( !(unsigned int)sub_716120(v9, v14) )
      goto LABEL_5;
    v17 = (const __m128i *)v26[0];
  }
  else
  {
    if ( *(_BYTE *)(v9 + 24) != 2 )
    {
LABEL_5:
      v15 = 0;
      goto LABEL_6;
    }
    v17 = *(const __m128i **)(v9 + 56);
  }
  v20 = v17;
  if ( !v17 )
    goto LABEL_5;
  if ( !(unsigned int)sub_7164A0(v8, (__int64)v25, v21, v6, a5) )
    goto LABEL_5;
  v18 = v20[10].m128i_i8[13];
  if ( v18 == 12 )
    goto LABEL_5;
  v19 = v25[10].m128i_i8[13];
  if ( v19 == 12 )
    goto LABEL_5;
  if ( !v19 || !v18 )
  {
    v15 = 1;
    sub_72C970(a2);
    goto LABEL_6;
  }
  sub_70F370(v25, *(unsigned __int8 *)(a1 + 56), v20, a2, &v24, &v23, &v22);
  if ( v24 )
    goto LABEL_5;
  v15 = 1;
  if ( v23 )
  {
    if ( v22 != 5 )
      goto LABEL_5;
  }
LABEL_6:
  sub_724E30(&v25);
  sub_724E30(v26);
  return v15;
}
