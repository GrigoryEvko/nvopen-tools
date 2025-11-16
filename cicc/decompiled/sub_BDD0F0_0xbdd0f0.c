// Function: sub_BDD0F0
// Address: 0xbdd0f0
//
void __fastcall sub_BDD0F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  const char *v5; // r13
  const char *v6; // r12
  unsigned int v7; // r15d
  _QWORD *v8; // rbx
  int v9; // eax
  __int64 v10; // rdx
  const char *v11; // r13
  const char *v12; // rbx
  __int64 v13; // rax
  const char *v14; // r15
  const char *v15; // r14
  const char *v16; // r13
  const char *v17; // rbx
  const char *v18; // r14
  __int64 v19; // rdx
  const char *v20; // r13
  const char *v21; // r12
  __int64 v22; // rbx
  const char *v23; // r15
  const char *v24; // rbx
  const char *v25; // r12
  const char *v26; // rbx
  const char *v27; // rax
  __int64 v28; // r12
  _BYTE *v29; // rax
  __int64 v30; // rax
  const char *v31; // [rsp+10h] [rbp-390h]
  int v32; // [rsp+24h] [rbp-37Ch]
  const char *v34; // [rsp+28h] [rbp-378h]
  _QWORD v35[4]; // [rsp+30h] [rbp-370h] BYREF
  char v36; // [rsp+50h] [rbp-350h]
  char v37; // [rsp+51h] [rbp-34Fh]
  const char *v38; // [rsp+60h] [rbp-340h] BYREF
  int v39; // [rsp+68h] [rbp-338h]
  _BYTE v40[16]; // [rsp+70h] [rbp-330h] BYREF
  char v41; // [rsp+80h] [rbp-320h]
  char v42; // [rsp+81h] [rbp-31Fh]

  v3 = *(_QWORD *)(a2 - 32);
  v4 = *(_QWORD *)(v3 + 56);
  sub_B428A0((__int64 *)&v38, (_BYTE *)v4, *(_QWORD *)(v3 + 64));
  v5 = v38;
  v32 = 0;
  v6 = &v38[192 * v39];
  if ( v38 == v6 )
  {
LABEL_33:
    if ( v6 != v40 )
      _libc_free(v6, v4);
    if ( *(_BYTE *)a2 == 40 )
    {
      if ( *(_DWORD *)(a2 + 88) == v32 )
        return;
      v42 = 1;
      v38 = "Number of label constraints does not match number of callbr dests";
      v41 = 3;
      sub_BDBF70(a1, (__int64)&v38);
      if ( !*a1 )
        return;
    }
    else
    {
      if ( !v32 )
        return;
      v42 = 1;
      v38 = "Label constraints can only be used with callbr";
      v41 = 3;
      sub_BDBF70(a1, (__int64)&v38);
      if ( !*a1 )
        return;
    }
    sub_BDBD80((__int64)a1, (_BYTE *)a2);
    return;
  }
  v7 = 0;
  v8 = (_QWORD *)(a2 + 72);
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = *(_DWORD *)v5;
      if ( *(_DWORD *)v5 == 3 )
      {
        ++v32;
        goto LABEL_4;
      }
      if ( !v9 )
        break;
      if ( v9 == 1 && v5[10] )
        goto LABEL_8;
LABEL_4:
      v5 += 192;
      if ( v6 == v5 )
        goto LABEL_11;
    }
    if ( !v5[10] )
    {
      v4 = v7;
      if ( (unsigned __int8)sub_B49B80(a2, v7, 82) )
      {
        v4 = (__int64)v35;
        v37 = 1;
        v35[0] = "Elementtype attribute can only be applied for indirect constraints";
        v36 = 3;
        sub_BDBF70(a1, (__int64)v35);
        if ( !*a1 )
          goto LABEL_42;
        goto LABEL_41;
      }
      goto LABEL_10;
    }
LABEL_8:
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 32 * (v7 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL)
                  + 8LL) != 14 )
    {
      v37 = 1;
      v27 = "Operand for indirect constraint must have pointer type";
      goto LABEL_72;
    }
    v4 = v7;
    if ( !sub_A74920(v8, v7) )
      break;
LABEL_10:
    v5 += 192;
    ++v7;
    if ( v6 == v5 )
    {
LABEL_11:
      v31 = v38;
      v6 = &v38[192 * v39];
      if ( v38 != v6 )
      {
        do
        {
          v10 = *((unsigned int *)v6 - 30);
          v11 = (const char *)*((_QWORD *)v6 - 16);
          v6 -= 192;
          v12 = &v11[56 * v10];
          if ( v11 != v12 )
          {
            do
            {
              v13 = *((unsigned int *)v12 - 10);
              v14 = (const char *)*((_QWORD *)v12 - 6);
              v12 -= 56;
              v13 *= 32;
              v15 = &v14[v13];
              if ( v14 != &v14[v13] )
              {
                do
                {
                  v15 -= 32;
                  if ( *(const char **)v15 != v15 + 16 )
                  {
                    v4 = *((_QWORD *)v15 + 2) + 1LL;
                    j_j___libc_free_0(*(_QWORD *)v15, v4);
                  }
                }
                while ( v14 != v15 );
                v14 = (const char *)*((_QWORD *)v12 + 1);
              }
              if ( v14 != v12 + 24 )
                _libc_free(v14, v4);
            }
            while ( v11 != v12 );
            v11 = (const char *)*((_QWORD *)v6 + 8);
          }
          if ( v11 != v6 + 80 )
            _libc_free(v11, v4);
          v16 = (const char *)*((_QWORD *)v6 + 2);
          v17 = &v16[32 * *((unsigned int *)v6 + 6)];
          if ( v16 != v17 )
          {
            do
            {
              v17 -= 32;
              if ( *(const char **)v17 != v17 + 16 )
              {
                v4 = *((_QWORD *)v17 + 2) + 1LL;
                j_j___libc_free_0(*(_QWORD *)v17, v4);
              }
            }
            while ( v16 != v17 );
            v16 = (const char *)*((_QWORD *)v6 + 2);
          }
          if ( v16 != v6 + 32 )
            _libc_free(v16, v4);
        }
        while ( v31 != v6 );
        v6 = v38;
      }
      goto LABEL_33;
    }
  }
  v37 = 1;
  v27 = "Operand for indirect constraint must have elementtype attribute";
LABEL_72:
  v35[0] = v27;
  v36 = 3;
  v28 = *a1;
  if ( *a1 )
  {
    v4 = *a1;
    sub_CA0E80(v35, *a1);
    v29 = *(_BYTE **)(v28 + 32);
    if ( (unsigned __int64)v29 >= *(_QWORD *)(v28 + 24) )
    {
      v4 = 10;
      sub_CB5D20(v28, 10);
    }
    else
    {
      *(_QWORD *)(v28 + 32) = v29 + 1;
      *v29 = 10;
    }
    v30 = *a1;
    *((_BYTE *)a1 + 152) = 1;
    if ( v30 )
    {
LABEL_41:
      v4 = a2;
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
    }
  }
  else
  {
    *((_BYTE *)a1 + 152) = 1;
  }
LABEL_42:
  v34 = v38;
  v18 = &v38[192 * v39];
  if ( v38 != v18 )
  {
    do
    {
      v19 = *((unsigned int *)v18 - 30);
      v20 = (const char *)*((_QWORD *)v18 - 16);
      v18 -= 192;
      v21 = &v20[56 * v19];
      if ( v20 != v21 )
      {
        do
        {
          v22 = *((unsigned int *)v21 - 10);
          v23 = (const char *)*((_QWORD *)v21 - 6);
          v21 -= 56;
          v24 = &v23[32 * v22];
          if ( v23 != v24 )
          {
            do
            {
              v24 -= 32;
              if ( *(const char **)v24 != v24 + 16 )
              {
                v4 = *((_QWORD *)v24 + 2) + 1LL;
                j_j___libc_free_0(*(_QWORD *)v24, v4);
              }
            }
            while ( v23 != v24 );
            v23 = (const char *)*((_QWORD *)v21 + 1);
          }
          if ( v23 != v21 + 24 )
            _libc_free(v23, v4);
        }
        while ( v20 != v21 );
        v20 = (const char *)*((_QWORD *)v18 + 8);
      }
      if ( v20 != v18 + 80 )
        _libc_free(v20, v4);
      v25 = (const char *)*((_QWORD *)v18 + 2);
      v26 = &v25[32 * *((unsigned int *)v18 + 6)];
      if ( v25 != v26 )
      {
        do
        {
          v26 -= 32;
          if ( *(const char **)v26 != v26 + 16 )
          {
            v4 = *((_QWORD *)v26 + 2) + 1LL;
            j_j___libc_free_0(*(_QWORD *)v26, v4);
          }
        }
        while ( v25 != v26 );
        v25 = (const char *)*((_QWORD *)v18 + 2);
      }
      if ( v25 != v18 + 32 )
        _libc_free(v25, v4);
    }
    while ( v34 != v18 );
    v18 = v38;
  }
  if ( v18 != v40 )
    _libc_free(v18, v4);
}
