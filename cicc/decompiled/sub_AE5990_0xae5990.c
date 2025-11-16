// Function: sub_AE5990
// Address: 0xae5990
//
unsigned int *__fastcall sub_AE5990(unsigned int *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // r15
  char v8; // bl
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 *v15; // rsi
  __int64 v16; // rax
  int v17; // edx
  bool v18; // cc
  unsigned int v19; // r15d
  __int64 v20; // rdi
  char *v21; // rsi
  unsigned __int64 v22; // rdx
  __int64 v23; // r9
  unsigned int v24; // eax
  __int64 v25; // rdi
  unsigned int v26; // eax
  bool v27; // zf
  char *v29; // r15
  __int64 v30; // r15
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  char v35; // [rsp+30h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 4;
  *((_QWORD *)a1 + 1) = 0x300000000LL;
  v7 = *a3;
  v8 = sub_AE5020(a2, *a3);
  v9 = sub_9208B0(a2, v7);
  v34 = v10;
  v33 = v9;
  LOBYTE(v32) = v10;
  sub_AE1360((__int64)&v33, ((1LL << v8) + ((unsigned __int64)(v9 + 7) >> 3) - 1) >> v8 << v8, v32, (__int64 *)a4);
  v11 = a1[2];
  v12 = v11 + 1;
  if ( v11 + 1 > (unsigned __int64)a1[3] )
  {
    v30 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 > (unsigned __int64)&v33 || (unsigned __int64)&v33 >= v30 + 16 * v11 )
    {
      sub_AE4800(a1, v12);
      v11 = a1[2];
      v13 = *(_QWORD *)a1;
      v15 = &v33;
      v14 = a1[2];
    }
    else
    {
      sub_AE4800(a1, v12);
      v13 = *(_QWORD *)a1;
      v11 = a1[2];
      v15 = (__int64 *)((char *)&v33 + *(_QWORD *)a1 - v30);
      v14 = a1[2];
    }
  }
  else
  {
    v13 = *(_QWORD *)a1;
    v14 = a1[2];
    v15 = &v33;
  }
  v16 = v13 + 16 * v11;
  if ( v16 )
  {
    v17 = *((_DWORD *)v15 + 2);
    *((_DWORD *)v15 + 2) = 0;
    *(_DWORD *)(v16 + 8) = v17;
    *(_QWORD *)v16 = *v15;
    v14 = a1[2];
  }
  v18 = (unsigned int)v34 <= 0x40;
  a1[2] = v14 + 1;
  if ( !v18 )
    goto LABEL_6;
LABEL_8:
  v19 = *(_DWORD *)(a4 + 8);
  if ( v19 > 0x40 )
  {
    while ( v19 - (unsigned int)sub_C444A0(a4) > 0x40 || **(_QWORD **)a4 )
    {
LABEL_10:
      sub_AE5800((__int64)&v33, a2, a3, (unsigned __int64 **)a4);
      if ( !v35 )
        return a1;
      v20 = a1[2];
      v21 = (char *)&v33;
      v22 = *(_QWORD *)a1;
      v23 = v20 + 1;
      v24 = a1[2];
      if ( v20 + 1 > (unsigned __int64)a1[3] )
      {
        if ( v22 > (unsigned __int64)&v33 || (unsigned __int64)&v33 >= v22 + 16 * v20 )
        {
          sub_AE4800(a1, v23);
          v20 = a1[2];
          v22 = *(_QWORD *)a1;
          v21 = (char *)&v33;
          v24 = a1[2];
        }
        else
        {
          v29 = (char *)&v33 - v22;
          sub_AE4800(a1, v23);
          v22 = *(_QWORD *)a1;
          v20 = a1[2];
          v21 = &v29[*(_QWORD *)a1];
          v24 = a1[2];
        }
      }
      v25 = v22 + 16 * v20;
      if ( v25 )
      {
        v26 = *((_DWORD *)v21 + 2);
        *(_DWORD *)(v25 + 8) = v26;
        if ( v26 > 0x40 )
          sub_C43780(v25, v21);
        else
          *(_QWORD *)v25 = *(_QWORD *)v21;
        v24 = a1[2];
      }
      v27 = v35 == 0;
      a1[2] = v24 + 1;
      if ( v27 )
        goto LABEL_8;
      v35 = 0;
      if ( (unsigned int)v34 > 0x40 )
      {
LABEL_6:
        if ( v33 )
          j_j___libc_free_0_0(v33);
        goto LABEL_8;
      }
      v19 = *(_DWORD *)(a4 + 8);
      if ( v19 <= 0x40 )
        goto LABEL_9;
    }
  }
  else
  {
LABEL_9:
    if ( *(_QWORD *)a4 )
      goto LABEL_10;
  }
  return a1;
}
