// Function: sub_2E7B0A0
// Address: 0x2e7b0a0
//
int *__fastcall sub_2E7B0A0(
        __int64 *a1,
        const void *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9)
{
  bool v13; // r12
  bool v14; // r11
  bool v15; // r10
  bool v16; // r8
  bool v17; // al
  bool v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rax
  int *v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v25; // rax
  bool v26; // [rsp+Eh] [rbp-42h]
  bool v27; // [rsp+Fh] [rbp-41h]
  int *v29; // [rsp+18h] [rbp-38h]

  v13 = a4 != 0;
  v14 = a5 != 0;
  v15 = a6 != 0;
  v16 = a9 != 0;
  v17 = a6 != 0;
  v18 = a8 != 0;
  v19 = 4LL * (a8 != 0) + 16 + 8 * ((((a9 != 0) + (a7 != 0) + v17) & 7) + a3 + ((v14 + (a4 != 0)) & 3));
  v20 = *a1;
  a1[10] += v19;
  v21 = (int *)((v20 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( a1[1] >= (unsigned __int64)v21 + v19 && v20 )
  {
    *a1 = (__int64)v21 + v19;
  }
  else
  {
    v26 = v15;
    v27 = v14;
    v25 = sub_9D1E70((__int64)a1, v19, v19, 3);
    v18 = a8 != 0;
    v16 = a9 != 0;
    v15 = v26;
    v14 = v27;
    v21 = (int *)v25;
  }
  *v21 = a3;
  *((_BYTE *)v21 + 4) = v13;
  *((_BYTE *)v21 + 5) = v14;
  *((_BYTE *)v21 + 6) = v15;
  *((_BYTE *)v21 + 7) = a7 != 0;
  *((_BYTE *)v21 + 8) = v18;
  *((_BYTE *)v21 + 9) = v16;
  if ( 8 * a3 )
  {
    v29 = v21;
    memmove(v21 + 4, a2, 8 * a3);
    v21 = v29;
  }
  if ( a4 )
    *(_QWORD *)&v21[2 * (int)a3 + 4] = a4;
  if ( a5 )
    *(_QWORD *)&v21[2 * v13 + 4 + 2 * *v21] = a5;
  v22 = 0;
  if ( a6 )
  {
    *(_QWORD *)&v21[2 * *v21 + 4 + 2 * (__int64)(*((unsigned __int8 *)v21 + 5) + *((unsigned __int8 *)v21 + 4))] = a6;
    v22 = 1;
  }
  v23 = (unsigned int)v22;
  if ( a7 )
  {
    v23 = (unsigned int)(v22 + 1);
    *(_QWORD *)&v21[2 * v22
                  + 4
                  + 2 * *v21
                  + 2 * (__int64)(*((unsigned __int8 *)v21 + 5) + *((unsigned __int8 *)v21 + 4))] = a7;
  }
  if ( a8 )
    v21[2 * *v21
      + 4
      + 2 * *((unsigned __int8 *)v21 + 7)
      + 2 * *((unsigned __int8 *)v21 + 6)
      + 2 * (__int64)(*((unsigned __int8 *)v21 + 5) + *((unsigned __int8 *)v21 + 4))] = a8;
  if ( a9 )
    *(_QWORD *)&v21[2 * v23
                  + 4
                  + 2 * *v21
                  + 2 * (__int64)(*((unsigned __int8 *)v21 + 5) + *((unsigned __int8 *)v21 + 4))] = a9;
  return v21;
}
