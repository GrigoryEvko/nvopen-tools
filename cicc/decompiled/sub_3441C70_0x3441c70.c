// Function: sub_3441C70
// Address: 0x3441c70
//
__int64 __fastcall sub_3441C70(unsigned int **a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ecx
  unsigned int **v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r13d
  __int64 v11; // rdx
  unsigned int *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 v16; // rdx
  unsigned int *v17; // rax
  __int64 v18; // rax
  unsigned int *v19; // rdi
  unsigned int *v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // r15
  unsigned int *v23; // rax
  bool (__fastcall *v24)(__int64, __int64, __int64, __int64, __int64, int, __int64, __int64, int); // r10
  __int64 v25; // r9
  __int64 v26; // rsi
  __int64 (*v27)(); // rax
  __int64 v29; // rax
  unsigned int v30; // r13d
  __int64 v31; // rax
  unsigned int v32; // r12d
  bool v33; // al

  v6 = 1;
  v7 = a1;
  v8 = *(_QWORD *)(a2 + 56);
  if ( !v8 )
    goto LABEL_17;
  do
  {
    while ( a3 != *(_DWORD *)(v8 + 8) )
    {
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_9;
    }
    if ( !v6 )
      goto LABEL_17;
    v9 = *(_QWORD *)(v8 + 32);
    if ( !v9 )
      goto LABEL_10;
    if ( a3 == *(_DWORD *)(v9 + 8) )
      goto LABEL_17;
    v8 = *(_QWORD *)(v9 + 32);
    v6 = 0;
  }
  while ( v8 );
LABEL_9:
  if ( v6 == 1 )
    goto LABEL_17;
LABEL_10:
  v10 = *(_DWORD *)(a2 + 24);
  if ( v10 == 190 )
  {
    **a1 = 192;
  }
  else
  {
    if ( v10 != 192 )
    {
LABEL_17:
      LODWORD(v7) = 0;
      return (unsigned int)v7;
    }
    **a1 = 190;
  }
  v11 = *(_QWORD *)(a2 + 40);
  v12 = a1[2];
  *(_QWORD *)v12 = *(_QWORD *)v11;
  v12[2] = *(_DWORD *)(v11 + 8);
  v15 = sub_33DFBC0(*(_QWORD *)a1[2], *((_QWORD *)a1[2] + 1), 1u, 1u, a5, a6);
  if ( !v15 )
    goto LABEL_17;
  v16 = *(_QWORD *)(a2 + 40);
  v17 = a1[3];
  *(_QWORD *)v17 = *(_QWORD *)(v16 + 40);
  v17[2] = *(_DWORD *)(v16 + 48);
  v18 = sub_33DFBC0(*(_QWORD *)a1[1], *((_QWORD *)a1[1] + 1), 1u, 1u, v13, v14);
  v19 = a1[5];
  v20 = v7[3];
  v21 = v18;
  v22 = **v7;
  v23 = v7[1];
  v24 = *(bool (__fastcall **)(__int64, __int64, __int64, __int64, __int64, int, __int64, __int64, int))(*(_QWORD *)v19 + 432LL);
  if ( v24 != sub_2FE3E80 )
  {
    LODWORD(v7) = ((__int64 (__fastcall *)(unsigned int *, _QWORD, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64, unsigned int *))v24)(
                    v19,
                    *(_QWORD *)v23,
                    *((_QWORD *)v23 + 1),
                    v21,
                    v15,
                    v10,
                    *(_QWORD *)v20,
                    *((_QWORD *)v20 + 1),
                    v22,
                    v7[4]);
    return (unsigned int)v7;
  }
  v25 = *(_QWORD *)v23;
  v26 = *((_QWORD *)v23 + 1);
  v27 = *(__int64 (**)())(*(_QWORD *)v19 + 400LL);
  if ( v27 == sub_2FE3030
    || !((unsigned __int8 (__fastcall *)(unsigned int *, __int64, __int64, _QWORD, _QWORD))v27)(
          v19,
          v25,
          v26,
          *(_QWORD *)v20,
          *((_QWORD *)v20 + 1)) )
  {
    goto LABEL_16;
  }
  if ( v10 != 190
    || ((v31 = *(_QWORD *)(v15 + 96), v32 = *(_DWORD *)(v31 + 32), v32 <= 0x40)
      ? (v33 = *(_QWORD *)(v31 + 24) == 1)
      : (v33 = v32 - 1 == (unsigned int)sub_C444A0(v31 + 24)),
        LODWORD(v7) = 0,
        !v33) )
  {
    LOBYTE(v7) = (_DWORD)v22 == 190 && v21 != 0;
    if ( (_BYTE)v7 )
    {
      v29 = *(_QWORD *)(v21 + 96);
      v30 = *(_DWORD *)(v29 + 32);
      if ( v30 <= 0x40 )
      {
        if ( *(_QWORD *)(v29 + 24) == 1 )
          return (unsigned int)v7;
      }
      else if ( (unsigned int)sub_C444A0(v29 + 24) == v30 - 1 )
      {
        return (unsigned int)v7;
      }
    }
LABEL_16:
    LOBYTE(v7) = v21 == 0;
  }
  return (unsigned int)v7;
}
