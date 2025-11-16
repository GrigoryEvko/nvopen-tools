// Function: sub_2720090
// Address: 0x2720090
//
void __fastcall sub_2720090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rcx
  _QWORD *v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdi
  _QWORD *v22; // rdx
  _QWORD *v23; // rdx
  unsigned __int64 v24; // rdi
  _QWORD *v25; // rcx
  int v26; // r13d
  int v27; // eax
  unsigned __int64 v28[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 24;
  v8 = *(unsigned int *)(a1 + 16);
  v9 = *(_QWORD *)(a1 + 8);
  while ( 1 )
  {
    v8 *= 3;
    v10 = v9 + 8 * v8 - 24;
    v11 = *(_QWORD *)(v10 + 8);
    if ( *(_QWORD *)v10 == v11 )
      break;
    v12 = *(_QWORD *)(v11 + 8);
    for ( *(_QWORD *)(v10 + 8) = v12; v12; *(_QWORD *)(v10 + 8) = v12 )
    {
      v8 = (unsigned int)**(unsigned __int8 **)(v12 + 24) - 30;
      if ( (unsigned __int8)(**(_BYTE **)(v12 + 24) - 30) <= 0xAu )
        break;
      v12 = *(_QWORD *)(v12 + 8);
    }
    v13 = *(_QWORD *)a1;
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 40LL);
    if ( !*(_BYTE *)(*(_QWORD *)a1 + 28LL) )
      goto LABEL_14;
    v15 = *(_QWORD **)(v13 + 8);
    v10 = *(unsigned int *)(v13 + 20);
    v8 = (__int64)&v15[v10];
    if ( v15 != (_QWORD *)v8 )
    {
      while ( v14 != *v15 )
      {
        if ( (_QWORD *)v8 == ++v15 )
          goto LABEL_34;
      }
      v8 = *(unsigned int *)(a1 + 16);
      goto LABEL_11;
    }
LABEL_34:
    if ( (unsigned int)v10 < *(_DWORD *)(v13 + 16) )
    {
      *(_DWORD *)(v13 + 20) = v10 + 1;
      *(_QWORD *)v8 = v14;
      ++*(_QWORD *)v13;
      LODWORD(v8) = *(_DWORD *)(a1 + 16);
LABEL_15:
      v17 = *(_QWORD *)(v14 + 16);
      if ( v17 )
      {
        while ( (unsigned __int8)(**(_BYTE **)(v17 + 24) - 30) > 0xAu )
        {
          v17 = *(_QWORD *)(v17 + 8);
          if ( !v17 )
          {
            v18 = (unsigned int)v8;
            if ( *(_DWORD *)(a1 + 20) > (unsigned int)v8 )
              goto LABEL_18;
            goto LABEL_23;
          }
        }
      }
      v18 = (unsigned int)v8;
      if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v8 )
      {
LABEL_23:
        v20 = sub_C8D7D0(a1 + 8, v7, 0, 0x18u, v28, a6);
        v9 = v20;
        v21 = 3LL * *(unsigned int *)(a1 + 16);
        v22 = (_QWORD *)(v21 * 8 + v20);
        if ( v21 * 8 + v20 )
        {
          *v22 = 0;
          v22[1] = v17;
          v22[2] = v14;
          v21 = 3LL * *(unsigned int *)(a1 + 16);
        }
        v23 = *(_QWORD **)(a1 + 8);
        v24 = (unsigned __int64)&v23[v21];
        if ( v23 != (_QWORD *)v24 )
        {
          v25 = (_QWORD *)v20;
          do
          {
            if ( v25 )
            {
              *v25 = *v23;
              v25[1] = v23[1];
              v25[2] = v23[2];
            }
            v23 += 3;
            v25 += 3;
          }
          while ( (_QWORD *)v24 != v23 );
          v24 = *(_QWORD *)(a1 + 8);
        }
        v26 = v28[0];
        if ( v7 != v24 )
          _libc_free(v24);
        v27 = *(_DWORD *)(a1 + 16);
        *(_QWORD *)(a1 + 8) = v9;
        *(_DWORD *)(a1 + 20) = v26;
        v8 = (unsigned int)(v27 + 1);
        *(_DWORD *)(a1 + 16) = v8;
      }
      else
      {
LABEL_18:
        v9 = *(_QWORD *)(a1 + 8);
        v19 = (_QWORD *)(v9 + 24 * v18);
        if ( v19 )
        {
          *v19 = 0;
          v19[1] = v17;
          v19[2] = v14;
          LODWORD(v8) = *(_DWORD *)(a1 + 16);
          v9 = *(_QWORD *)(a1 + 8);
        }
        v8 = (unsigned int)(v8 + 1);
        *(_DWORD *)(a1 + 16) = v8;
      }
    }
    else
    {
LABEL_14:
      sub_C8CC70(v13, v14, v8, v10, a5, a6);
      a5 = v16;
      v8 = *(unsigned int *)(a1 + 16);
      if ( (_BYTE)a5 )
        goto LABEL_15;
LABEL_11:
      v9 = *(_QWORD *)(a1 + 8);
    }
  }
}
