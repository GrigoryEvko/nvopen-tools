// Function: sub_80D480
// Address: 0x80d480
//
__int64 __fastcall sub_80D480(__int64 a1, __int64 a2, const char *a3)
{
  const char *v4; // r12
  size_t v5; // rax
  size_t v6; // rbx
  size_t v7; // r14
  __int64 v8; // r15
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // rbx
  signed __int64 v14; // rsi
  __int64 v15; // r15
  _BYTE *v16; // rdx
  _BYTE *v17; // rdx
  __int64 v18; // r14
  _BYTE *v19; // rdx
  _BYTE *v20; // r9
  _BYTE *v21; // rcx
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 v24; // r12
  _BYTE *v25; // rcx
  _BYTE *v26; // rdx
  _BYTE *v27; // rcx
  _BYTE *v28; // [rsp+8h] [rbp-A8h]
  _BYTE *v29; // [rsp+10h] [rbp-A0h]
  _BYTE *v30; // [rsp+10h] [rbp-A0h]
  _BYTE *v31; // [rsp+18h] [rbp-98h]
  _BYTE v32[56]; // [rsp+28h] [rbp-88h] BYREF
  _BYTE *v33; // [rsp+60h] [rbp-50h]
  __int64 v34; // [rsp+68h] [rbp-48h]
  __int64 v35; // [rsp+70h] [rbp-40h]

  v4 = a3;
  v5 = strlen(a3);
  v33 = 0;
  v6 = v5 + 1;
  v34 = 0;
  v7 = v5;
  v8 = v5 + 1;
  v35 = 0;
  if ( v5 + 1 > 0x32 )
  {
    v10 = (_BYTE *)sub_823970(v5 + 1);
    v9 = *(_QWORD *)(a1 + 64);
    v34 = v6;
    v33 = v10;
    v31 = (_BYTE *)(a1 + 8);
    if ( v9 == a1 + 8 )
    {
LABEL_4:
      v11 = v35;
      *(_DWORD *)(a1 + 4) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_QWORD *)(a1 + 80) = v11;
      if ( v10 != v32 )
        goto LABEL_5;
      goto LABEL_47;
    }
LABEL_3:
    sub_823A00(v9, *(_QWORD *)(a1 + 72));
    v10 = v33;
    v8 = v34;
    goto LABEL_4;
  }
  v9 = *(_QWORD *)(a1 + 64);
  v33 = v32;
  v34 = v5 + 1;
  v31 = (_BYTE *)(a1 + 8);
  if ( v9 != a1 + 8 )
    goto LABEL_3;
  *(_DWORD *)(a1 + 4) = 0;
  v11 = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
LABEL_47:
  if ( v8 <= 50 )
  {
    *(_DWORD *)(a1 + 4) = 1;
    v10 = v31;
  }
  else
  {
    v10 = (_BYTE *)sub_823970(v8);
  }
  v26 = v10;
  v27 = v32;
  if ( v11 > 0 )
  {
    do
    {
      if ( v26 )
        *v26 = *v27;
      ++v26;
      ++v27;
    }
    while ( &v10[v11] != v26 );
  }
LABEL_5:
  *(_QWORD *)(a1 + 72) = v8;
  *(_QWORD *)(a1 + 64) = v10;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  result = sub_823A00(0, 0);
  v13 = *(_QWORD *)(a1 + 80);
  v14 = *(_QWORD *)(a1 + 72);
  v15 = v7 + v13;
  if ( (__int64)(v7 + v13) <= v14 )
  {
    v16 = *(_BYTE **)(a1 + 64);
    goto LABEL_7;
  }
  result = *(unsigned int *)(a1 + 4);
  v20 = *(_BYTE **)(a1 + 64);
  if ( (!(_DWORD)result || v20 == v31) && v15 <= 50 )
  {
    *(_DWORD *)(a1 + 4) = 1;
    v16 = v31;
  }
  else
  {
    v28 = *(_BYTE **)(a1 + 64);
    result = sub_823970(v7 + v13);
    v20 = v28;
    v16 = (_BYTE *)result;
  }
  if ( v20 == v16 )
    goto LABEL_29;
  result = (__int64)v16;
  v21 = v20;
  if ( v13 > 0 )
  {
    do
    {
      if ( result )
        *(_BYTE *)result = *v21;
      ++result;
      ++v21;
    }
    while ( (_BYTE *)result != &v16[v13] );
  }
  if ( v20 != v31 )
  {
    v29 = v16;
    result = sub_823A00(v20, v14);
    v16 = v29;
LABEL_29:
    *(_QWORD *)(a1 + 64) = v16;
    *(_QWORD *)(a1 + 72) = v15;
    goto LABEL_7;
  }
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 64) = v16;
  *(_QWORD *)(a1 + 72) = v15;
LABEL_7:
  if ( v7 )
  {
    result = (__int64)&v16[v13];
    v17 = &v16[v15];
    do
    {
      if ( result )
        *(_BYTE *)result = *v4;
      ++result;
      ++v4;
    }
    while ( v17 != (_BYTE *)result );
    v16 = *(_BYTE **)(a1 + 64);
  }
  v18 = *(_QWORD *)(a1 + 80) + v7;
  *(_QWORD *)(a1 + 80) = v18;
  if ( *(_QWORD *)(a1 + 72) == v18 )
  {
    result = *(unsigned int *)(a1 + 4);
    if ( v18 <= 1 )
    {
      if ( (_DWORD)result && v16 != v31 )
      {
        v23 = 2;
        v22 = 2;
        goto LABEL_33;
      }
      v22 = 2;
    }
    else
    {
      v22 = v18 + (v18 >> 1) + 1;
      if ( (_DWORD)result )
      {
        v23 = v18 + (v18 >> 1) + 1;
        if ( v16 != v31 )
          goto LABEL_33;
      }
      if ( v22 > 50 )
      {
        v23 = v18 + (v18 >> 1) + 1;
LABEL_33:
        v30 = v16;
        result = sub_823970(v23);
        v16 = v30;
        v24 = result;
LABEL_34:
        if ( v16 != (_BYTE *)v24 )
        {
          result = v24;
          v25 = v16;
          if ( v18 > 0 )
          {
            do
            {
              if ( result )
                *(_BYTE *)result = *v25;
              ++result;
              ++v25;
            }
            while ( v18 + v24 != result );
          }
          if ( v16 == v31 )
            *(_DWORD *)(a1 + 4) = 0;
          else
            result = sub_823A00(v16, v18);
        }
        *(_QWORD *)(a1 + 64) = v24;
        v16 = (_BYTE *)v24;
        *(_QWORD *)(a1 + 72) = v22;
        goto LABEL_14;
      }
    }
    *(_DWORD *)(a1 + 4) = 1;
    v24 = (__int64)v31;
    goto LABEL_34;
  }
LABEL_14:
  v19 = &v16[v18];
  if ( v19 )
    *v19 = 0;
  *(_QWORD *)(a1 + 80) = v18 + 1;
  return result;
}
