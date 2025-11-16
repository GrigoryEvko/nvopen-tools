// Function: sub_D5D330
// Address: 0xd5d330
//
__int64 __fastcall sub_D5D330(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // r15
  int v7; // esi
  int v8; // edx
  char *v9; // rax
  __int64 v10; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+4h] [rbp-4Ch] BYREF
  int v19; // [rsp+Ch] [rbp-44h]
  __m128i v20; // [rsp+10h] [rbp-40h] BYREF
  int v21; // [rsp+24h] [rbp-2Ch] BYREF
  char v22; // [rsp+28h] [rbp-28h]

  v5 = sub_D5BAA0((unsigned __int8 *)a2);
  if ( !v5 )
    goto LABEL_10;
  if ( !a3 )
    goto LABEL_10;
  v6 = v5;
  if ( !sub_981210(*a3, v5, &v17) )
    goto LABEL_10;
  v7 = v17;
  if ( (a3[((unsigned __int64)v17 >> 6) + 1] & (1LL << v17)) != 0
    || (((int)*(unsigned __int8 *)(*a3 + (v17 >> 2)) >> (2 * (v17 & 3))) & 3) == 0 )
  {
    goto LABEL_10;
  }
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v6 + 24) + 16LL) + 8LL) == 14 )
  {
    sub_D5BC90(&v20, v6, 7u, a3);
    if ( v22 )
    {
      v15 = sub_D5CA30(&v21);
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v15;
      *(_QWORD *)(a1 + 8) = v16;
      return a1;
    }
    v7 = v17;
  }
  else
  {
    v22 = 0;
  }
  v18 = sub_D5D290(v6, v7);
  v19 = v8;
  if ( !(_BYTE)v8 )
  {
LABEL_10:
    if ( (sub_D5BB80((unsigned __int8 *)a2) & 7) == 0 )
    {
LABEL_11:
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
    v20.m128i_i64[0] = *(_QWORD *)(a2 + 72);
    v12 = sub_A747B0(&v20, -1, "alloc-family", 0xCu);
    if ( v12 )
    {
      v20.m128i_i64[0] = v12;
    }
    else
    {
      v20.m128i_i64[0] = sub_B49600(a2, "alloc-family", 0xCu);
      if ( !v20.m128i_i64[0] )
        goto LABEL_11;
    }
    v13 = sub_A72240(v20.m128i_i64);
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v13;
    *(_QWORD *)(a1 + 8) = v14;
    return a1;
  }
  v9 = sub_D5CA30((_DWORD *)&v18 + 1);
  *(_BYTE *)(a1 + 16) = 1;
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 8) = v10;
  return a1;
}
