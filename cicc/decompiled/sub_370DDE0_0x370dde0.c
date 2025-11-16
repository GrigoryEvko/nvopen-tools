// Function: sub_370DDE0
// Address: 0x370dde0
//
unsigned __int64 *__fastcall sub_370DDE0(unsigned __int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  bool v5; // zf
  __int64 v6; // r15
  __int64 v7; // rbx
  unsigned __int64 v8; // rbx
  int v9; // r10d
  __int16 v10; // si
  __int16 v11; // ax
  __int16 v12; // bx
  int v13; // r14d
  __int64 v14; // r8
  __int64 v15; // r9
  int v17; // r12d
  unsigned int *v18; // rdx
  __int64 v19; // rax
  unsigned int *v20; // r13
  unsigned int *v21; // r14
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  unsigned int v24; // ebx
  __int64 v25; // rsi
  int v26; // ebx
  __int16 v27; // r12
  unsigned int (*v28)(void); // rax
  unsigned int *v29; // rdx
  unsigned int *v30; // r13
  unsigned int *v31; // rbx
  __int16 v32; // bx
  __int16 v33; // [rsp+Eh] [rbp-D2h]
  _QWORD *v34; // [rsp+28h] [rbp-B8h]
  unsigned int v35; // [rsp+3Ch] [rbp-A4h] BYREF
  unsigned __int64 v36; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 v37; // [rsp+48h] [rbp-98h] BYREF
  __m128i v38[2]; // [rsp+50h] [rbp-90h] BYREF
  char v39; // [rsp+70h] [rbp-70h]
  char v40; // [rsp+71h] [rbp-6Fh]
  __int64 v41[4]; // [rsp+80h] [rbp-60h] BYREF
  char v42; // [rsp+A0h] [rbp-40h]
  char v43; // [rsp+A1h] [rbp-3Fh]

  v5 = a2[9] == 0;
  v40 = 1;
  v34 = a2 + 2;
  v6 = a2[7];
  v38[0].m128i_i64[0] = (__int64)"NumArgs";
  v39 = 3;
  if ( v5 )
  {
    v25 = a2[8];
    if ( v25 && !v6 )
    {
      v26 = *(_DWORD *)(a4 + 16);
      v27 = v26;
      v28 = *(unsigned int (**)(void))(**(_QWORD **)(v25 + 24) + 16LL);
      if ( (char *)v28 != (char *)sub_3700C70 )
      {
        v32 = __ROL2__(v26, 8);
        if ( v28() != 1 )
          v27 = v32;
      }
      LOWORD(v37) = v27;
      sub_3719260(v41, v25, &v37, 2);
      if ( (v41[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v36 = 0;
        v41[0] = v41[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_9C6670((__int64 *)&v36, v41);
        sub_9C66B0(v41);
        v8 = v36 & 0xFFFFFFFFFFFFFFFELL;
LABEL_22:
        if ( v8 )
          goto LABEL_11;
LABEL_28:
        *a1 = 1;
        return a1;
      }
      v41[0] = 0;
      sub_9C66B0(v41);
      v29 = *(unsigned int **)(a4 + 8);
      if ( v29 != &v29[*(unsigned int *)(a4 + 16)] )
      {
        v30 = &v29[*(unsigned int *)(a4 + 16)];
        v31 = v29;
        do
        {
          v43 = 1;
          v41[0] = (__int64)"Argument";
          v42 = 3;
          sub_37011E0(&v37, v34, v31, v41);
          v22 = v37 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v37 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_21;
          ++v31;
          v37 = 0;
          sub_9C66B0((__int64 *)&v37);
        }
        while ( v30 != v31 );
      }
LABEL_27:
      v41[0] = 0;
      v36 = 1;
      sub_9C66B0(v41);
      goto LABEL_28;
    }
  }
  else if ( !v6 && !a2[8] )
  {
    v17 = *(_DWORD *)(a4 + 16);
    sub_370BB40(v34, v38);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a2[9] + 8LL))(a2[9], (unsigned __int16)v17, 2);
    if ( a2[9] && !a2[7] && !a2[8] )
      a2[10] += 2LL;
    v18 = *(unsigned int **)(a4 + 8);
    v19 = *(unsigned int *)(a4 + 16);
    if ( v18 != &v18[v19] )
    {
      v20 = &v18[v19];
      v21 = v18;
      while ( 1 )
      {
        v43 = 1;
        v41[0] = (__int64)"Argument";
        v42 = 3;
        sub_37011E0(&v37, v34, v21, v41);
        v22 = v37 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v37 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
        if ( v20 == ++v21 )
          goto LABEL_27;
      }
LABEL_21:
      v36 = 0;
      v37 = v22 | 1;
      sub_9C6670((__int64 *)&v36, &v37);
      sub_9C66B0((__int64 *)&v37);
      v8 = v36 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_22;
    }
    goto LABEL_27;
  }
  v41[0] = 0;
  v41[1] = 0;
  sub_1254950(&v37, v6, (__int64)v41, 2u);
  v7 = v37;
  v37 = 0;
  v8 = v7 & 0xFFFFFFFFFFFFFFFELL;
  if ( !v8 )
  {
    sub_9C66B0((__int64 *)&v37);
    v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 24) + 16LL))(*(_QWORD *)(v6 + 24));
    v10 = *(_WORD *)v41[0];
    v37 = 0;
    v11 = __ROL2__(v10, 8);
    if ( v9 == 1 )
      v11 = v10;
    v12 = v11;
    v33 = v11;
    sub_9C66B0((__int64 *)&v37);
    v41[0] = 0;
    sub_9C66B0(v41);
    if ( v12 )
    {
      v13 = 0;
      while ( 1 )
      {
        v35 = 0;
        v43 = 1;
        v41[0] = (__int64)"Argument";
        v42 = 3;
        sub_37011E0(&v37, v34, &v35, v41);
        v8 = v37 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v37 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
        v23 = *(unsigned int *)(a4 + 16);
        v24 = v35;
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 20) )
        {
          sub_C8D5F0(a4 + 8, (const void *)(a4 + 24), v23 + 1, 4u, v14, v15);
          v23 = *(unsigned int *)(a4 + 16);
        }
        ++v13;
        *(_DWORD *)(*(_QWORD *)(a4 + 8) + 4 * v23) = v24;
        ++*(_DWORD *)(a4 + 16);
        if ( (_WORD)v13 == v33 )
          goto LABEL_27;
      }
      v37 = 0;
      v36 = v8 | 1;
      sub_9C66B0((__int64 *)&v37);
      goto LABEL_11;
    }
    goto LABEL_27;
  }
  sub_9C66B0((__int64 *)&v37);
  v41[0] = 0;
  v36 = v8 | 1;
  sub_9C66B0(v41);
LABEL_11:
  *a1 = v8 | 1;
  return a1;
}
