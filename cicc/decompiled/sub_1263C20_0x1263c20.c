// Function: sub_1263C20
// Address: 0x1263c20
//
void __fastcall sub_1263C20(char **a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r12
  bool v5; // zf
  char *v6; // r14
  char *v7; // r15
  char **v8; // r14
  char *v9; // rdx
  char *v10; // rcx
  char *v11; // rbx
  char v12; // r11
  char **v13; // r9
  _BYTE *v14; // r10
  char v15; // r14
  _BYTE *v16; // rax
  __int64 v17; // r15
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx
  _BYTE *v20; // rdi
  __int64 *v21; // rdi
  __int64 *v22; // rdi
  _BYTE *v23; // rdx
  __int64 v24; // rsi
  unsigned __int64 v25; // rcx
  char v26; // al
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 *v29; // rdi
  _BYTE *v33; // [rsp+18h] [rbp-68h]
  char **v34; // [rsp+20h] [rbp-60h]
  char v35; // [rsp+20h] [rbp-60h]
  char v36; // [rsp+28h] [rbp-58h]
  __int64 v37; // [rsp+28h] [rbp-58h]
  _BYTE *v38; // [rsp+30h] [rbp-50h] BYREF
  __int64 v39; // [rsp+38h] [rbp-48h]
  _QWORD v40[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = v40;
  v5 = a1[1] == 0;
  v6 = *a1;
  v38 = v40;
  v39 = 0;
  LOBYTE(v40[0]) = 0;
  if ( v5 )
    return;
  v7 = v6;
  v8 = a1;
  do
  {
    if ( sub_22417D0(a4, (unsigned int)*v7, 0) == -1 )
    {
      if ( sub_22417D0(a3, (unsigned int)*v7, 0) == -1 )
      {
        v23 = v38;
        v24 = v39;
        v25 = 15;
        v26 = *v7;
        if ( v38 != v4 )
          v25 = v40[0];
        v27 = v39 + 1;
        if ( v39 + 1 > v25 )
        {
          v35 = *v7;
          v37 = v39;
          sub_2240BB0(&v38, v39, 0, 0, 1);
          v23 = v38;
          v26 = v35;
          v24 = v37;
        }
        v23[v24] = v26;
        v39 = v27;
        v38[v24 + 1] = 0;
        v9 = *v8;
        v10 = v8[1];
      }
      else
      {
        v19 = v39;
        if ( v39 )
        {
          v22 = *(__int64 **)(a2 + 8);
          if ( v22 == *(__int64 **)(a2 + 16) )
          {
            sub_8FD760((__m128i **)a2, *(const __m128i **)(a2 + 8), (__int64)&v38);
            v19 = v39;
          }
          else
          {
            if ( v22 )
            {
              *v22 = (__int64)(v22 + 2);
              sub_12631D0(v22, v38, (__int64)&v38[v39]);
              v22 = *(__int64 **)(a2 + 8);
              v19 = v39;
            }
            *(_QWORD *)(a2 + 8) = v22 + 4;
          }
          sub_2241130(&v38, 0, v19, byte_3F871B3, 0);
        }
        v9 = *v8;
        v10 = v8[1];
      }
    }
    else
    {
      v9 = *v8;
      v10 = v8[1];
      v11 = v7 + 1;
      v12 = *v7;
      if ( v7 + 1 == &v10[(_QWORD)*v8] )
      {
LABEL_22:
        v21 = *(__int64 **)(a2 + 8);
        if ( v21 != *(__int64 **)(a2 + 16) )
        {
          if ( v21 )
          {
            *v21 = (__int64)(v21 + 2);
            sub_12631D0(v21, v38, (__int64)&v38[v39]);
            v21 = *(__int64 **)(a2 + 8);
          }
          *(_QWORD *)(a2 + 8) = v21 + 4;
          goto LABEL_17;
        }
        goto LABEL_42;
      }
      v13 = v8;
      v14 = v4;
      while ( 1 )
      {
        v15 = *v11;
        v7 = v11;
        if ( *v11 == v12 )
          break;
        v16 = v38;
        v17 = v39;
        v18 = 15;
        if ( v38 != v14 )
          v18 = v40[0];
        if ( v39 + 1 > v18 )
        {
          v33 = v14;
          v34 = v13;
          v36 = v12;
          sub_2240BB0(&v38, v39, 0, 0, 1);
          v16 = v38;
          v14 = v33;
          v13 = v34;
          v12 = v36;
        }
        v16[v17] = v15;
        ++v11;
        v39 = v17 + 1;
        v38[v17 + 1] = 0;
        v9 = *v13;
        v10 = v13[1];
        if ( &v10[(_QWORD)*v13] == v11 )
        {
          v4 = v14;
          goto LABEL_22;
        }
      }
      v8 = v13;
      v4 = v14;
    }
    ++v7;
  }
  while ( v7 != &v9[(_QWORD)v10] );
  if ( !v39 )
  {
LABEL_17:
    v20 = v38;
    goto LABEL_18;
  }
  v28 = *(_QWORD **)(a2 + 8);
  if ( v28 == *(_QWORD **)(a2 + 16) )
  {
LABEL_42:
    sub_8FD760((__m128i **)a2, *(const __m128i **)(a2 + 8), (__int64)&v38);
    goto LABEL_17;
  }
  if ( v28 )
  {
    v29 = *(__int64 **)(a2 + 8);
    *v28 = v28 + 2;
    sub_12631D0(v29, v38, (__int64)&v38[v39]);
    v28 = *(_QWORD **)(a2 + 8);
  }
  v20 = v38;
  *(_QWORD *)(a2 + 8) = v28 + 4;
LABEL_18:
  if ( v20 != v4 )
    j_j___libc_free_0(v20, v40[0] + 1LL);
}
