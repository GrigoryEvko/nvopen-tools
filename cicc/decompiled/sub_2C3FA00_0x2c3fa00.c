// Function: sub_2C3FA00
// Address: 0x2c3fa00
//
__int64 *__fastcall sub_2C3FA00(__int64 a1, __int64 a2, int a3)
{
  __int64 *v3; // rbx
  __int64 *result; // rax
  unsigned int v5; // r13d
  __int64 v6; // rdx
  __int64 v7; // r9
  int v8; // r14d
  unsigned int v9; // eax
  __int64 *v10; // rdi
  __int64 *v11; // rcx
  __int64 v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // r14
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r10
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r12
  int v26; // edi
  int v27; // eax
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 *v31; // [rsp+20h] [rbp-60h]
  __int64 v33; // [rsp+38h] [rbp-48h]
  __int64 v34; // [rsp+40h] [rbp-40h] BYREF
  __int64 v35[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = *(__int64 **)(a2 + 48);
  result = &v3[*(unsigned int *)(a2 + 56)];
  v31 = result;
  if ( v3 != result )
  {
    v5 = 0;
    v28 = (unsigned int)(a3 - 1);
    v33 = a2 + 40;
    while ( 1 )
    {
      v14 = *v3;
      v34 = *v3;
      if ( !a3 )
        goto LABEL_6;
      v23 = sub_2BF04A0(v14);
      v14 = v34;
      if ( !v23 )
        goto LABEL_6;
      v24 = *(_DWORD *)(a1 + 184);
      v25 = a1 + 160;
      if ( !v24 )
        break;
      v6 = v34;
      v7 = *(_QWORD *)(a1 + 168);
      v8 = 1;
      v9 = (v24 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v10 = (__int64 *)(v7 + 72LL * v9);
      v11 = 0;
      v12 = *v10;
      if ( v34 != *v10 )
      {
        while ( v12 != -4096 )
        {
          if ( !v11 && v12 == -8192 )
            v11 = v10;
          v9 = (v24 - 1) & (v8 + v9);
          v10 = (__int64 *)(v7 + 72LL * v9);
          v12 = *v10;
          if ( v34 == *v10 )
            goto LABEL_4;
          ++v8;
        }
        if ( !v11 )
          v11 = v10;
        ++*(_QWORD *)(a1 + 160);
        v27 = *(_DWORD *)(a1 + 176);
        v35[0] = (__int64)v11;
        v26 = v27 + 1;
        if ( 4 * (v27 + 1) < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(a1 + 180) - v26 <= v24 >> 3 )
          {
LABEL_18:
            sub_2C3F4B0(v25, v24);
            sub_2C3F050(v25, &v34, v35);
            v6 = v34;
            v11 = (__int64 *)v35[0];
            v26 = *(_DWORD *)(a1 + 176) + 1;
          }
          *(_DWORD *)(a1 + 176) = v26;
          if ( *v11 != -4096 )
            --*(_DWORD *)(a1 + 180);
          v13 = v11 + 3;
          *v11 = v6;
          v11[1] = (__int64)(v11 + 3);
          v11[2] = 0x600000000LL;
          goto LABEL_5;
        }
LABEL_17:
        v24 *= 2;
        goto LABEL_18;
      }
LABEL_4:
      v13 = (_QWORD *)v10[1];
LABEL_5:
      v14 = v13[v28];
LABEL_6:
      v15 = 8LL * v5;
      v16 = *(_QWORD *)(v15 + *(_QWORD *)(a2 + 48));
      v35[0] = v33;
      v17 = *(_QWORD **)(v16 + 16);
      v18 = (__int64)&v17[*(unsigned int *)(v16 + 24)];
      v19 = sub_2C3DA80(v17, v18, v35);
      if ( (_QWORD *)v18 != v19 )
      {
        if ( (_QWORD *)v18 != v19 + 1 )
        {
          memmove(v19, v19 + 1, v18 - (_QWORD)(v19 + 1));
          LODWORD(v21) = *(_DWORD *)(v16 + 24);
        }
        v21 = (unsigned int)(v21 - 1);
        *(_DWORD *)(v16 + 24) = v21;
        v22 = (__int64 *)(v15 + *(_QWORD *)(a2 + 48));
      }
      *v22 = v14;
      result = (__int64 *)*(unsigned int *)(v14 + 24);
      if ( (unsigned __int64)result + 1 > *(unsigned int *)(v14 + 28) )
      {
        sub_C8D5F0(v14 + 16, (const void *)(v14 + 32), (unsigned __int64)result + 1, 8u, v20, v21);
        result = (__int64 *)*(unsigned int *)(v14 + 24);
      }
      ++v5;
      ++v3;
      *(_QWORD *)(*(_QWORD *)(v14 + 16) + 8LL * (_QWORD)result) = v33;
      ++*(_DWORD *)(v14 + 24);
      if ( v31 == v3 )
        return result;
    }
    v35[0] = 0;
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_17;
  }
  return result;
}
