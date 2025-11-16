// Function: sub_B43120
// Address: 0xb43120
//
_QWORD *__fastcall sub_B43120(_QWORD *a1, __int64 a2, char *a3, __int64 a4, __int64 a5)
{
  const char *v7; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  int v14; // edi
  const char *v15; // rax
  int v16; // r10d
  int v17; // r11d
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rax
  _QWORD *v26; // r15
  _QWORD *v27; // r14
  _QWORD *v28; // r12
  _QWORD *v29; // rbx
  __int64 v30; // [rsp+0h] [rbp-350h]
  const char *v32; // [rsp+10h] [rbp-340h] BYREF
  unsigned int v33; // [rsp+18h] [rbp-338h]
  char v34; // [rsp+20h] [rbp-330h] BYREF

  if ( *(_DWORD *)(a2 + 8) >> 8 )
  {
    sub_B3B200((__int64)a1, "inline asm cannot be variadic", (__int64)a3, a4, a5);
    return a1;
  }
  v7 = a3;
  sub_B428A0((__int64 *)&v32, a3, a4);
  v12 = v33;
  if ( !v33 )
  {
    if ( a4 )
    {
      v7 = "failed to parse constraints";
      sub_B3B200((__int64)a1, "failed to parse constraints", v9, v10, v11);
      v12 = v33;
      goto LABEL_25;
    }
    v13 = *(_QWORD *)(a2 + 16);
    v14 = 0;
LABEL_7:
    if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) != 7 )
    {
      v7 = "inline asm without outputs must return void";
      sub_B3B200((__int64)a1, "inline asm without outputs must return void", v13, v10, v11);
      v12 = v33;
      goto LABEL_25;
    }
    goto LABEL_23;
  }
  v15 = v32;
  v11 = 0;
  v16 = 0;
  v10 = 0;
  v14 = 0;
  v17 = 0;
  v7 = &v32[192 * v33];
  do
  {
    while ( 1 )
    {
      v19 = *(unsigned int *)v15;
      if ( (_DWORD)v19 == 2 )
      {
        v10 = (unsigned int)(v10 + 1);
        goto LABEL_14;
      }
      if ( (unsigned int)v19 > 2 )
      {
        if ( (_DWORD)v19 == 3 )
        {
          if ( (_DWORD)v10 )
          {
            v7 = "label constraint occurs after clobber constraint";
            sub_B3B200((__int64)a1, "label constraint occurs after clobber constraint", v19, v10, v11);
            v12 = v33;
            goto LABEL_25;
          }
          v11 = (unsigned int)(v11 + 1);
        }
        goto LABEL_14;
      }
      if ( !(_DWORD)v19 )
        break;
      v18 = (unsigned int)v11 | (unsigned int)v10;
      if ( (_DWORD)v18 || v16 != v14 )
      {
        v7 = "output constraint occurs after input, clobber or label constraint";
        sub_B3B200((__int64)a1, "output constraint occurs after input, clobber or label constraint", v19, v18, v11);
        v12 = v33;
        goto LABEL_25;
      }
      if ( v15[10] )
      {
        ++v16;
        goto LABEL_19;
      }
      ++v17;
      v11 = 0;
      v10 = 0;
LABEL_14:
      v15 += 192;
      if ( v7 == v15 )
        goto LABEL_20;
    }
    if ( (_DWORD)v10 )
    {
      v7 = "input constraint occurs after clobber constraint";
      sub_B3B200((__int64)a1, "input constraint occurs after clobber constraint", v19, v10, v11);
      v12 = v33;
      goto LABEL_25;
    }
LABEL_19:
    v15 += 192;
    ++v14;
    v10 = 0;
  }
  while ( v7 != v15 );
LABEL_20:
  v13 = *(_QWORD *)(a2 + 16);
  v20 = *(_QWORD *)v13;
  if ( !v17 )
    goto LABEL_7;
  if ( v17 != 1 )
  {
    if ( *(_BYTE *)(v20 + 8) != 15 || v17 != *(_DWORD *)(v20 + 12) )
    {
      v7 = "number of output constraints does not match number of return struct elements";
      sub_B3B200(
        (__int64)a1,
        "number of output constraints does not match number of return struct elements",
        v13,
        v10,
        v11);
      v12 = v33;
      goto LABEL_25;
    }
LABEL_23:
    if ( *(_DWORD *)(a2 + 12) - 1 == v14 )
    {
      *a1 = 1;
    }
    else
    {
      v7 = "number of input constraints does not match number of parameters";
      sub_B3B200((__int64)a1, "number of input constraints does not match number of parameters", v13, v10, v11);
      v12 = v33;
    }
    goto LABEL_25;
  }
  if ( *(_BYTE *)(v20 + 8) != 15 )
    goto LABEL_23;
  v7 = "inline asm with one output cannot return struct";
  sub_B3B200((__int64)a1, "inline asm with one output cannot return struct", v13, v10, v11);
  v12 = v33;
LABEL_25:
  v21 = (__int64)&v32[192 * v12];
  v30 = (__int64)v32;
  if ( v32 != (const char *)v21 )
  {
    do
    {
      v22 = *(unsigned int *)(v21 - 120);
      v23 = *(_QWORD *)(v21 - 128);
      v21 -= 192;
      v24 = v23 + 56 * v22;
      if ( v23 != v24 )
      {
        do
        {
          v25 = *(unsigned int *)(v24 - 40);
          v26 = *(_QWORD **)(v24 - 48);
          v24 -= 56;
          v25 *= 32;
          v27 = (_QWORD *)((char *)v26 + v25);
          if ( v26 != (_QWORD *)((char *)v26 + v25) )
          {
            do
            {
              v27 -= 4;
              if ( (_QWORD *)*v27 != v27 + 2 )
              {
                v7 = (const char *)(v27[2] + 1LL);
                j_j___libc_free_0(*v27, v7);
              }
            }
            while ( v26 != v27 );
            v26 = *(_QWORD **)(v24 + 8);
          }
          if ( v26 != (_QWORD *)(v24 + 24) )
            _libc_free(v26, v7);
        }
        while ( v23 != v24 );
        v23 = *(_QWORD *)(v21 + 64);
      }
      if ( v23 != v21 + 80 )
        _libc_free(v23, v7);
      v28 = *(_QWORD **)(v21 + 16);
      v29 = &v28[4 * *(unsigned int *)(v21 + 24)];
      if ( v28 != v29 )
      {
        do
        {
          v29 -= 4;
          if ( (_QWORD *)*v29 != v29 + 2 )
          {
            v7 = (const char *)(v29[2] + 1LL);
            j_j___libc_free_0(*v29, v7);
          }
        }
        while ( v28 != v29 );
        v28 = *(_QWORD **)(v21 + 16);
      }
      if ( v28 != (_QWORD *)(v21 + 32) )
        _libc_free(v28, v7);
    }
    while ( v30 != v21 );
    v21 = (__int64)v32;
  }
  if ( (char *)v21 != &v34 )
    _libc_free(v21, v7);
  return a1;
}
