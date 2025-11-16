// Function: sub_159C590
// Address: 0x159c590
//
_QWORD *__fastcall sub_159C590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rdi
  _QWORD *result; // rax
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // [rsp+10h] [rbp-B0h]
  __int64 v35; // [rsp+10h] [rbp-B0h]
  __int64 v37; // [rsp+28h] [rbp-98h]
  __int64 v39; // [rsp+38h] [rbp-88h]
  __int64 v40; // [rsp+38h] [rbp-88h]
  __int64 v41; // [rsp+38h] [rbp-88h]
  __int64 v42; // [rsp+38h] [rbp-88h]
  __int64 v43; // [rsp+38h] [rbp-88h]
  __int64 v44; // [rsp+38h] [rbp-88h]
  _QWORD v45[2]; // [rsp+48h] [rbp-78h] BYREF
  __int64 v46; // [rsp+58h] [rbp-68h] BYREF
  __int64 v47; // [rsp+60h] [rbp-60h]
  __int64 v48; // [rsp+78h] [rbp-48h] BYREF
  __int64 v49; // [rsp+80h] [rbp-40h]

  v3 = a2;
  *(_QWORD *)(a1 + 16) = 0;
  v39 = sub_16982B0(a1, a2);
  v6 = sub_16982C0(a1, a2, v4, v5);
  if ( v39 == v6 )
    sub_169C630(&v48, v39, 1);
  else
    sub_1699170(&v48, v39, 1);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v7 + 40LL * *(unsigned int *)(a1 + 24);
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_6;
      v9 = v7 + 8;
      if ( v6 == v48 )
      {
        sub_169C6E0(v9, &v48);
        v7 += 40;
        if ( v8 == v7 )
          break;
      }
      else
      {
        sub_16986C0(v9, &v48);
LABEL_6:
        v7 += 40;
        if ( v8 == v7 )
          break;
      }
    }
  }
  if ( v6 == v48 )
  {
    v25 = v49;
    if ( v49 )
    {
      v26 = 32LL * *(_QWORD *)(v49 - 8);
      v27 = v49 + v26;
      if ( v49 != v49 + v26 )
      {
        do
        {
          v27 -= 32;
          v37 = v25;
          sub_127D120((_QWORD *)(v27 + 8));
          v25 = v37;
        }
        while ( v37 != v27 );
      }
      j_j_j___libc_free_0_0(v25 - 8);
    }
  }
  else
  {
    sub_1698460(&v48);
  }
  if ( v39 == v6 )
  {
    sub_169C630(&v46, v6, 1);
    sub_169C630(&v48, v6, 2);
  }
  else
  {
    sub_1699170(&v46, v39, 1);
    sub_1699170(&v48, v39, 2);
  }
  result = v45;
  v11 = (__int64 *)(v3 + 8);
  if ( v3 != a3 )
  {
    while ( 1 )
    {
      v12 = *v11;
      if ( *v11 == v46 )
      {
        if ( v6 == v12 )
          result = (_QWORD *)sub_169CB90(v11, &v46);
        else
          result = (_QWORD *)sub_1698510(v11, &v46);
        if ( (_BYTE)result )
          goto LABEL_24;
        v12 = *v11;
        if ( v48 != *v11 )
          break;
      }
      else if ( v48 != v12 )
      {
        break;
      }
      result = (_QWORD *)(v6 == v12 ? sub_169CB90(v11, &v48) : sub_1698510(v11, &v48));
      if ( !(_BYTE)result )
        break;
      if ( v6 == *v11 )
      {
LABEL_44:
        v16 = v11[1];
        if ( v16 )
        {
          v17 = 32LL * *(_QWORD *)(v16 - 8);
          v18 = v16 + v17;
          if ( v16 != v16 + v17 )
          {
            do
            {
              v41 = v18 - 32;
              sub_127D120((_QWORD *)(v18 - 24));
              v18 = v41;
            }
            while ( v16 != v41 );
          }
          result = (_QWORD *)j_j_j___libc_free_0_0(v16 - 8);
        }
        goto LABEL_26;
      }
LABEL_25:
      result = (_QWORD *)sub_1698460(v11);
LABEL_26:
      v3 += 40;
      v11 += 5;
      if ( a3 == v3 )
        goto LABEL_27;
    }
    sub_1598370(a1, v3, v45);
    v13 = v45[0];
    v14 = v45[0] + 8LL;
    if ( v6 == *(_QWORD *)(v45[0] + 8LL) )
    {
      if ( v6 == *v11 )
      {
        if ( (__int64 *)v14 != v11 )
        {
          v31 = *(_QWORD *)(v45[0] + 16LL);
          if ( v31 )
          {
            v32 = 32LL * *(_QWORD *)(v31 - 8);
            v33 = v31 + v32;
            if ( v31 != v31 + v32 )
            {
              do
              {
                v35 = v14;
                v43 = v33 - 32;
                sub_127D120((_QWORD *)(v33 - 24));
                v33 = v43;
                v14 = v35;
              }
              while ( v31 != v43 );
            }
            v44 = v14;
            j_j_j___libc_free_0_0(v31 - 8);
            v14 = v44;
          }
          sub_169C7E0(v14, v11);
        }
        goto LABEL_20;
      }
    }
    else if ( v6 != *v11 )
    {
      sub_16983E0(v45[0] + 8LL, v11);
LABEL_20:
      *(_QWORD *)(v13 + 32) = v11[3];
      result = (_QWORD *)a1;
      v11[3] = 0;
      ++*(_DWORD *)(a1 + 16);
      v15 = v11[3];
      if ( v15 )
      {
        if ( v6 == *(_QWORD *)(v15 + 32) )
        {
          v28 = *(_QWORD *)(v15 + 40);
          if ( v28 )
          {
            v29 = 32LL * *(_QWORD *)(v28 - 8);
            v30 = v28 + v29;
            if ( v28 != v28 + v29 )
            {
              do
              {
                v34 = v28;
                v42 = v30 - 32;
                sub_127D120((_QWORD *)(v30 - 24));
                v30 = v42;
                v28 = v34;
              }
              while ( v34 != v42 );
            }
            j_j_j___libc_free_0_0(v28 - 8);
          }
        }
        else
        {
          sub_1698460(v15 + 32);
        }
        sub_164BE60(v15);
        result = (_QWORD *)sub_1648B90(v15);
      }
LABEL_24:
      if ( v6 == *v11 )
        goto LABEL_44;
      goto LABEL_25;
    }
    if ( (__int64 *)v14 != v11 )
    {
      v40 = v45[0] + 8LL;
      sub_127D120((_QWORD *)(v45[0] + 8LL));
      if ( v6 == *v11 )
        sub_169C7E0(v40, v11);
      else
        sub_1698450(v40, v11);
    }
    goto LABEL_20;
  }
LABEL_27:
  if ( v6 == v48 )
  {
    v22 = v49;
    if ( v49 )
    {
      v23 = 32LL * *(_QWORD *)(v49 - 8);
      v24 = v49 + v23;
      if ( v49 != v49 + v23 )
      {
        do
        {
          v24 -= 32;
          sub_127D120((_QWORD *)(v24 + 8));
        }
        while ( v22 != v24 );
      }
      result = (_QWORD *)j_j_j___libc_free_0_0(v22 - 8);
    }
  }
  else
  {
    result = (_QWORD *)sub_1698460(&v48);
  }
  if ( v6 != v46 )
    return (_QWORD *)sub_1698460(&v46);
  v19 = v47;
  if ( v47 )
  {
    v20 = 32LL * *(_QWORD *)(v47 - 8);
    v21 = v47 + v20;
    if ( v47 != v47 + v20 )
    {
      do
      {
        v21 -= 32;
        sub_127D120((_QWORD *)(v21 + 8));
      }
      while ( v19 != v21 );
    }
    return (_QWORD *)j_j_j___libc_free_0_0(v19 - 8);
  }
  return result;
}
