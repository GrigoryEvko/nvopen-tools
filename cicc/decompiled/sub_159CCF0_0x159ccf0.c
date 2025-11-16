// Function: sub_159CCF0
// Address: 0x159ccf0
//
__int64 __fastcall sub_159CCF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 *v4; // rdi
  char v5; // al
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // rsi
  int v11; // eax
  int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // r13
  __int64 v24; // r12
  char v25; // al
  char v26; // dl
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h] BYREF
  __int64 v35; // [rsp+30h] [rbp-40h]

  v2 = a2;
  v3 = *a1;
  v4 = (__int64 *)(*a1 + 168LL);
  v5 = sub_1598370((__int64)v4, a2, &v33);
  v8 = v33;
  if ( v5 )
  {
    result = *(_QWORD *)(v33 + 32);
    if ( result )
      return result;
    goto LABEL_15;
  }
  v10 = *(unsigned int *)(v3 + 192);
  v11 = *(_DWORD *)(v3 + 184);
  ++*(_QWORD *)(v3 + 168);
  v12 = v11 + 1;
  if ( 4 * v12 >= (unsigned int)(3 * v10) )
  {
    v10 = (unsigned int)(2 * v10);
  }
  else
  {
    v13 = (unsigned int)v10 >> 3;
    if ( (int)v10 - *(_DWORD *)(v3 + 188) - v12 > (unsigned int)v13 )
      goto LABEL_6;
  }
  sub_159CB40(v3 + 168, v10);
  v10 = v2;
  v13 = v3 + 168;
  sub_1598370(v3 + 168, v2, &v33);
  v8 = v33;
  v12 = *(_DWORD *)(v3 + 184) + 1;
LABEL_6:
  *(_DWORD *)(v3 + 184) = v12;
  v14 = sub_16982B0(v13, v10);
  v17 = sub_16982C0(v13, v10, v15, v16);
  v18 = v17;
  v4 = &v34;
  if ( v14 == v17 )
    sub_169C630(&v34, v17, 1);
  else
    sub_1699170(&v34, v14, 1);
  v19 = *(_QWORD *)(v8 + 8);
  v20 = v34;
  if ( v19 != v34
    || ((v4 = (__int64 *)(v8 + 8), v19 == v18) ? (v25 = sub_169CB90(v4, &v34)) : (v25 = sub_1698510(v4, &v34)),
        v26 = v25,
        v20 = v34,
        !v26) )
  {
    --*(_DWORD *)(v3 + 188);
    if ( v18 == v20 )
      goto LABEL_33;
LABEL_10:
    v4 = &v34;
    sub_1698460(&v34);
    goto LABEL_11;
  }
  if ( v18 != v34 )
    goto LABEL_10;
LABEL_33:
  v6 = v35;
  if ( v35 )
  {
    v27 = 32LL * *(_QWORD *)(v35 - 8);
    v28 = v35 + v27;
    if ( v35 != v35 + v27 )
    {
      do
      {
        v29 = v6;
        v30 = v28 - 32;
        sub_127D120((_QWORD *)(v28 - 24));
        v28 = v30;
        v6 = v29;
      }
      while ( v29 != v30 );
    }
    v4 = (__int64 *)(v6 - 8);
    j_j_j___libc_free_0_0(v6 - 8);
  }
LABEL_11:
  v21 = *(_QWORD *)(v2 + 8);
  a2 = v2 + 8;
  if ( *(_QWORD *)(v8 + 8) == v18 )
  {
    if ( v21 == v18 )
    {
      v4 = (__int64 *)(v8 + 8);
      sub_16A0170(v8 + 8, a2);
      goto LABEL_14;
    }
LABEL_26:
    if ( v8 + 8 != a2 )
    {
      sub_127D120((_QWORD *)(v8 + 8));
      a2 = v2 + 8;
      v4 = (__int64 *)(v8 + 8);
      if ( v18 == *(_QWORD *)(v2 + 8) )
        sub_169C6E0(v4, a2);
      else
        sub_16986C0(v4, a2);
    }
    goto LABEL_14;
  }
  if ( v21 == v18 )
    goto LABEL_26;
  v4 = (__int64 *)(v8 + 8);
  sub_1698680(v8 + 8, a2);
LABEL_14:
  *(_QWORD *)(v8 + 32) = 0;
LABEL_15:
  v22 = *(_QWORD *)(v2 + 8);
  if ( v22 == sub_1698260(v4, a2, v6, v7) )
  {
    v23 = sub_1643290(a1);
  }
  else if ( v22 == sub_1698270(v4, a2) )
  {
    v23 = sub_16432A0(a1);
  }
  else if ( v22 == sub_1698280(v4) )
  {
    v23 = sub_16432B0(a1);
  }
  else if ( v22 == sub_16982A0() )
  {
    v23 = sub_16432E0(a1);
  }
  else if ( v22 == sub_1698290() )
  {
    v23 = sub_16432F0(a1);
  }
  else
  {
    v23 = sub_1643300(a1);
  }
  result = sub_1648A60(56, 0);
  if ( result )
  {
    v32 = result;
    sub_15940C0(result, v23, v2);
    result = v32;
  }
  v24 = *(_QWORD *)(v8 + 32);
  *(_QWORD *)(v8 + 32) = result;
  if ( v24 )
  {
    sub_127D120((_QWORD *)(v24 + 32));
    sub_164BE60(v24);
    sub_1648B90(v24);
    return *(_QWORD *)(v8 + 32);
  }
  return result;
}
