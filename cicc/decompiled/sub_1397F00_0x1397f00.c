// Function: sub_1397F00
// Address: 0x1397f00
//
void __fastcall sub_1397F00(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rdx
  __int64 v4; // r8
  unsigned int v5; // eax
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  char *v10; // r15
  __int64 *v11; // r14
  char *i; // rbx
  __int64 v13; // rax
  size_t v14; // rdx
  size_t v15; // r12
  const void *v16; // rsi
  size_t v17; // rdx
  const void *v18; // rdi
  size_t v19; // rcx
  int v20; // eax
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 **v23; // rbx
  __int64 *v24; // rdi
  char *v25; // r13
  char *v27; // [rsp+18h] [rbp-C8h]
  __int64 v28; // [rsp+18h] [rbp-C8h]
  size_t v29; // [rsp+18h] [rbp-C8h]
  __int64 v30; // [rsp+18h] [rbp-C8h]
  size_t v31; // [rsp+18h] [rbp-C8h]
  char *v32; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+28h] [rbp-B8h]
  _BYTE v34[176]; // [rsp+30h] [rbp-B0h] BYREF

  v2 = a1 + 16;
  v3 = *(_QWORD *)(a1 + 48);
  v32 = v34;
  v33 = 0x1000000000LL;
  if ( v3 > 0x10 )
  {
    sub_16CD150(&v32, v34, v3, 8);
    v4 = *(_QWORD *)(a1 + 32);
    v6 = (unsigned int)v33;
    if ( v4 != v2 )
      goto LABEL_4;
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 32);
    v5 = 16;
    v6 = 0;
    if ( v4 == v2 )
    {
      v25 = v34;
      goto LABEL_27;
    }
    while ( 1 )
    {
      v7 = *(_QWORD *)(v4 + 40);
      if ( (unsigned int)v6 >= v5 )
      {
        v30 = v4;
        sub_16CD150(&v32, v34, 0, 8);
        v6 = (unsigned int)v33;
        v4 = v30;
      }
      *(_QWORD *)&v32[8 * v6] = v7;
      v6 = (unsigned int)(v33 + 1);
      LODWORD(v33) = v33 + 1;
      v4 = sub_220EF30(v4);
      if ( v4 == v2 )
        break;
LABEL_4:
      v5 = HIDWORD(v33);
    }
  }
  v8 = 8 * v6;
  v25 = &v32[v8];
  if ( v32 != &v32[v8] )
  {
    v27 = v32;
    _BitScanReverse64(&v9, v8 >> 3);
    sub_1396FB0((__int64)v32, (void ***)&v32[v8], 2LL * (int)(63 - (v9 ^ 0x3F)));
    if ( (unsigned __int64)v8 <= 0x80 )
    {
      sub_1396B10(v27, v25);
      goto LABEL_24;
    }
    v10 = v27 + 128;
    sub_1396B10(v27, v27 + 128);
    if ( v25 != v27 + 128 )
    {
LABEL_11:
      while ( 2 )
      {
        v11 = *(__int64 **)v10;
        for ( i = v10; ; i -= 8 )
        {
          v21 = (__int64 *)*((_QWORD *)i - 1);
          v22 = *v21;
          if ( !*v21 )
          {
LABEL_21:
            v10 += 8;
            *(_QWORD *)i = v11;
            if ( v25 != v10 )
              goto LABEL_11;
            goto LABEL_24;
          }
          v28 = *v11;
          if ( *v11 )
            break;
          if ( !v22 )
          {
            v10 += 8;
            *(_QWORD *)i = v11;
            if ( v25 != v10 )
              goto LABEL_11;
            goto LABEL_24;
          }
LABEL_19:
          *(_QWORD *)i = v21;
        }
        v13 = sub_1649960(v22);
        v15 = v14;
        v16 = (const void *)v13;
        v18 = (const void *)sub_1649960(v28);
        v19 = v17;
        if ( v17 > v15 )
        {
          if ( !v15 )
            goto LABEL_21;
          v31 = v17;
          v20 = memcmp(v18, v16, v15);
          v19 = v31;
          if ( !v20 )
            goto LABEL_17;
        }
        else if ( !v17 || (v29 = v17, v20 = memcmp(v18, v16, v17), v19 = v29, !v20) )
        {
          if ( v19 == v15 )
            goto LABEL_21;
LABEL_17:
          if ( v19 >= v15 )
            goto LABEL_21;
LABEL_18:
          v21 = (__int64 *)*((_QWORD *)i - 1);
          goto LABEL_19;
        }
        if ( v20 >= 0 )
        {
          v10 += 8;
          *(_QWORD *)i = v11;
          if ( v25 != v10 )
            continue;
          break;
        }
        goto LABEL_18;
      }
    }
LABEL_24:
    v23 = (__int64 **)v32;
    v25 = &v32[8 * (unsigned int)v33];
    if ( v32 != v25 )
    {
      do
      {
        v24 = *v23++;
        sub_1397A10(v24, a2);
      }
      while ( v25 != (char *)v23 );
      v25 = v32;
    }
  }
LABEL_27:
  if ( v25 != v34 )
    _libc_free((unsigned __int64)v25);
}
