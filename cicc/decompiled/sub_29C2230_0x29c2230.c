// Function: sub_29C2230
// Address: 0x29c2230
//
void __fastcall sub_29C2230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 v7; // rcx
  _QWORD *v8; // rbx
  __int64 v9; // r15
  __int64 *i; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rax
  _QWORD *v16; // r12
  __int64 v17; // rax
  int v18; // r15d
  unsigned __int64 *v19; // r14
  unsigned __int64 *v20; // r15
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rcx
  unsigned __int64 *v24; // r15
  unsigned __int64 *v25; // rbx
  unsigned __int64 *v26; // r12
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // [rsp-58h] [rbp-58h]
  unsigned __int64 v30; // [rsp-58h] [rbp-58h]
  unsigned __int64 v31; // [rsp-58h] [rbp-58h]
  unsigned int v32; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v33; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(_QWORD **)a1;
    v32 = *(_DWORD *)(a2 + 8);
    v6 = v32;
    if ( v32 <= v7 )
    {
      v13 = *(_QWORD **)a1;
      if ( v32 )
      {
        v19 = *(unsigned __int64 **)a2;
        v29 = 32LL * v32;
        v20 = v8 + 1;
        v21 = (unsigned __int64 *)(*(_QWORD *)a2 + v29);
        do
        {
          v22 = v20[2];
          *(v20 - 1) = *v19;
          v23 = v19[3];
          if ( v22 != v23 )
          {
            if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
            {
              sub_BD60C0(v20);
              v23 = v19[3];
            }
            v20[2] = v23;
            if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
              sub_BD6050(v20, v19[1] & 0xFFFFFFFFFFFFFFF8LL);
          }
          v19 += 4;
          v20 += 4;
        }
        while ( v19 != v21 );
        v13 = *(_QWORD **)a1;
        v7 = *(unsigned int *)(a1 + 8);
        v8 = (_QWORD *)((char *)v8 + v29);
      }
      v14 = &v13[4 * v7];
      while ( v8 != v14 )
      {
        v15 = *(v14 - 1);
        v14 -= 4;
        if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
          sub_BD60C0(v14 + 1);
      }
    }
    else
    {
      if ( v32 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v16 = &v8[4 * v7];
        while ( v16 != v8 )
        {
          while ( 1 )
          {
            v17 = *(v16 - 1);
            v16 -= 4;
            if ( v17 == 0 || v17 == -4096 || v17 == -8192 )
              break;
            sub_BD60C0(v16 + 1);
            if ( v16 == v8 )
              goto LABEL_24;
          }
        }
LABEL_24:
        *(_DWORD *)(a1 + 8) = 0;
        v8 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, v32, 0x20u, &v33, a6);
        sub_29C2140(a1, v8);
        v18 = v33;
        if ( a1 + 16 != *(_QWORD *)a1 )
          _libc_free(*(_QWORD *)a1);
        *(_QWORD *)a1 = v8;
        v7 = 0;
        *(_DWORD *)(a1 + 12) = v18;
        v6 = *(unsigned int *)(a2 + 8);
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v24 = *(unsigned __int64 **)a2;
        v7 *= 32LL;
        v25 = v8 + 1;
        v26 = (unsigned __int64 *)(*(_QWORD *)a2 + v7);
        do
        {
          v27 = v25[2];
          *(v25 - 1) = *v24;
          v28 = v24[3];
          if ( v27 != v28 )
          {
            if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
            {
              v30 = v7;
              sub_BD60C0(v25);
              v28 = v24[3];
              v7 = v30;
            }
            v25[2] = v28;
            if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
            {
              v31 = v7;
              sub_BD6050(v25, v24[1] & 0xFFFFFFFFFFFFFFF8LL);
              v7 = v31;
            }
          }
          v24 += 4;
          v25 += 4;
        }
        while ( v24 != v26 );
        v6 = *(unsigned int *)(a2 + 8);
        v8 = (_QWORD *)(v7 + *(_QWORD *)a1);
      }
      v9 = *(_QWORD *)a2 + 32 * v6;
      for ( i = (__int64 *)(v7 + *(_QWORD *)a2); (__int64 *)v9 != i; v8 += 4 )
      {
        if ( v8 )
        {
          v11 = *i;
          v8[1] = 4;
          v8[2] = 0;
          *v8 = v11;
          v12 = i[3];
          v8[3] = v12;
          if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
            sub_BD6050(v8 + 1, i[1] & 0xFFFFFFFFFFFFFFF8LL);
        }
        i += 4;
      }
    }
    *(_DWORD *)(a1 + 8) = v32;
  }
}
