// Function: sub_16755D0
// Address: 0x16755d0
//
void __fastcall sub_16755D0(__int64 a1)
{
  _QWORD *v1; // rsi
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  __int64 v4; // r14
  __int64 **v5; // r15
  const char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  __int64 v11; // r15
  const char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  __int64 v17; // r15
  const char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 *v22; // r15
  __int64 v23; // r13
  __int64 v24; // r14
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // r9
  unsigned __int64 j; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 i; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 *v35; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h]

  v1 = *(_QWORD **)(a1 + 8);
  v2 = (_QWORD *)v1[2];
  v3 = v1 + 1;
  if ( v2 != v1 + 1 )
  {
    v4 = a1 + 48;
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      if ( (*((_BYTE *)v2 - 33) & 0x20) == 0 )
        goto LABEL_3;
      if ( (*(_BYTE *)(v2 - 3) & 0xFu) - 7 <= 1 )
        goto LABEL_3;
      v5 = *(__int64 ***)a1;
      v6 = sub_1649960((__int64)(v2 - 7));
      v8 = sub_1632000((__int64)v5, (__int64)v6, v7);
      if ( !v8 || (*(_BYTE *)(v8 + 32) & 0xFu) - 7 <= 1 )
        goto LABEL_3;
      if ( (*(_BYTE *)(v8 + 32) & 0xF) == 6 && (*(_BYTE *)(v2 - 3) & 0xF) == 6 )
      {
        sub_16711C0(v4, *(_QWORD *)(*(_QWORD *)(v8 + 24) + 24LL), *(_QWORD *)(*(v2 - 4) + 24LL));
LABEL_3:
        v2 = (_QWORD *)v2[1];
        if ( v3 == v2 )
          goto LABEL_12;
      }
      else
      {
        sub_16711C0(v4, *(_QWORD *)v8, *(v2 - 7));
        v2 = (_QWORD *)v2[1];
        if ( v3 == v2 )
        {
LABEL_12:
          v1 = *(_QWORD **)(a1 + 8);
          break;
        }
      }
    }
  }
  v9 = (_QWORD *)v1[4];
  v10 = v1 + 3;
  if ( v9 != v1 + 3 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v9 )
          BUG();
        if ( (*((_BYTE *)v9 - 33) & 0x20) != 0 && (*(_BYTE *)(v9 - 3) & 0xFu) - 7 > 1 )
        {
          v11 = *(_QWORD *)a1;
          v12 = sub_1649960((__int64)(v9 - 7));
          v14 = sub_1632000(v11, (__int64)v12, v13);
          if ( v14 )
          {
            if ( (*(_BYTE *)(v14 + 32) & 0xFu) - 7 > 1 )
              break;
          }
        }
        v9 = (_QWORD *)v9[1];
        if ( v10 == v9 )
          goto LABEL_22;
      }
      sub_16711C0(a1 + 48, *(_QWORD *)v14, *(v9 - 7));
      v9 = (_QWORD *)v9[1];
    }
    while ( v10 != v9 );
LABEL_22:
    v1 = *(_QWORD **)(a1 + 8);
  }
  v15 = (_QWORD *)v1[6];
  v16 = v1 + 5;
  if ( v1 + 5 != v15 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v15 )
          BUG();
        if ( (*((_BYTE *)v15 - 25) & 0x20) != 0 && (*(_BYTE *)(v15 - 2) & 0xFu) - 7 > 1 )
        {
          v17 = *(_QWORD *)a1;
          v18 = sub_1649960((__int64)(v15 - 6));
          v20 = sub_1632000(v17, (__int64)v18, v19);
          if ( v20 )
          {
            if ( (*(_BYTE *)(v20 + 32) & 0xFu) - 7 > 1 )
              break;
          }
        }
        v15 = (_QWORD *)v15[1];
        if ( v16 == v15 )
          goto LABEL_32;
      }
      sub_16711C0(a1 + 48, *(_QWORD *)v20, *(v15 - 6));
      v15 = (_QWORD *)v15[1];
    }
    while ( v16 != v15 );
LABEL_32:
    v1 = *(_QWORD **)(a1 + 8);
  }
  sub_1633130(&v35, (__int64)v1);
  v21 = v36;
  v22 = v35;
  for ( i = a1 + 48; v21 != v22; ++v22 )
  {
    v23 = *v22;
    if ( *(_QWORD *)(*v22 + 24) && !sub_16707E0(*(_QWORD *)(a1 + 688), *v22) )
    {
      v24 = sub_1643640(v23);
      v26 = v25;
      v27 = v25;
      for ( j = v25; j; --j )
      {
        v29 = j - 1;
        if ( *(_BYTE *)(j + v24 - 1) == 46 )
        {
          if ( j > 1 && *(_BYTE *)(v24 + v26 - 1) != 46 && (unsigned int)*(unsigned __int8 *)(v24 + j) - 48 <= 9 )
          {
            if ( v26 <= v29 )
              v29 = v26;
            v27 = v29;
          }
          break;
        }
      }
      v34 = v27;
      sub_1643640(v23);
      if ( v30 != v34 )
      {
        v31 = sub_1643CD0(*(__int64 ***)a1, v24, v34);
        v32 = v31;
        if ( v31 )
        {
          if ( sub_16707E0(*(_QWORD *)(a1 + 688), v31) )
            sub_16711C0(i, v32, v23);
        }
      }
    }
  }
  sub_16750C0(i);
  if ( v35 )
    j_j___libc_free_0(v35, v37 - (_QWORD)v35);
}
