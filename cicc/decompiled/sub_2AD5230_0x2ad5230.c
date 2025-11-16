// Function: sub_2AD5230
// Address: 0x2ad5230
//
__int64 __fastcall sub_2AD5230(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // r10d
  unsigned int i; // eax
  _QWORD *v10; // rdi
  unsigned int v11; // eax
  _BYTE *v12; // rsi
  __int64 v13; // r14
  unsigned __int64 v15; // rax
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rbx
  _QWORD *v24; // r13
  __int64 *v25; // r12
  int v26; // [rsp+4h] [rbp-9Ch]
  __int64 v27; // [rsp+8h] [rbp-98h]
  __int64 v28; // [rsp+10h] [rbp-90h]
  __int64 v29; // [rsp+18h] [rbp-88h]
  unsigned int v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+28h] [rbp-78h] BYREF
  __int64 v32[2]; // [rsp+30h] [rbp-70h] BYREF
  void *v33[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v6 = *(unsigned int *)(a1 + 88);
  v32[0] = a2;
  v32[1] = a3;
  v7 = *(_QWORD *)(a1 + 72);
  if ( !(_DWORD)v6 )
    goto LABEL_7;
  v8 = 1;
  for ( i = (v6 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v6 - 1) & v11 )
  {
    v10 = (_QWORD *)(v7 + 24LL * i);
    if ( a2 == *v10 && a3 == v10[1] )
      break;
    if ( *v10 == -4096 && v10[1] == -4096 )
      goto LABEL_7;
    v11 = v8 + i;
    ++v8;
  }
  if ( v10 != (_QWORD *)(v7 + 24 * v6) )
    return v10[2];
LABEL_7:
  v28 = a1 + 64;
  v12 = (_BYTE *)sub_986580(a2);
  if ( *v12 == 32 )
  {
    sub_2AD3FB0(a1, (__int64)v12);
    return *sub_2AD3ED0(a1 + 64, v32);
  }
  else
  {
    v13 = sub_2AB6F10(a1, a2);
    v15 = sub_986580(a2);
    v16 = v15;
    if ( *(_BYTE *)v15 != 31 )
      BUG();
    if ( (*(_DWORD *)(v15 + 4) & 0x7FFFFFF) == 3 && *(_QWORD *)(v15 - 64) != *(_QWORD *)(v15 - 32) )
    {
      v17 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v17 == a2 + 48 )
        goto LABEL_25;
      if ( !v17 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 > 0xA )
        goto LABEL_25;
      v29 = *(_QWORD *)(a1 + 8);
      v26 = sub_B46E30(v17 - 24);
      if ( !v26 )
        goto LABEL_25;
      v18 = 0;
      v27 = v29 + 56;
      while ( 1 )
      {
        v30 = v18;
        v19 = sub_B46EC0(v16, v18);
        if ( !(unsigned __int8)sub_B19060(v27, v19, v20, v21) )
          break;
        v18 = v30 + 1;
        if ( v30 + 1 == v26 )
          goto LABEL_25;
      }
      v22 = *(_QWORD *)(a1 + 32);
      if ( *(_BYTE *)(v22 + 664) )
      {
        if ( *(_QWORD *)(v22 + 648) == a2 )
        {
LABEL_25:
          v23 = sub_2AC59D0(a1, *(_BYTE **)(v16 - 96));
          if ( *(_QWORD *)(v16 - 32) != a3 )
          {
            v24 = *(_QWORD **)(a1 + 56);
            v34 = 257;
            v31 = *(_QWORD *)(v16 + 48);
            if ( v31 )
              sub_2AAAFA0(&v31);
            v23 = sub_2AB0C10(v24, v23, &v31, v33);
            sub_9C6650(&v31);
          }
          if ( v13 )
          {
            v25 = *(__int64 **)(a1 + 56);
            v34 = 257;
            v31 = *(_QWORD *)(v16 + 48);
            if ( v31 )
              sub_2AAAFA0(&v31);
            v13 = sub_2AB1320(v25, v13, v23, &v31, v33);
            sub_9C6650(&v31);
          }
          else
          {
            v13 = v23;
          }
        }
      }
    }
    *sub_2AD3ED0(v28, v32) = v13;
  }
  return v13;
}
