// Function: sub_2EFF3E0
// Address: 0x2eff3e0
//
__int64 **__fastcall sub_2EFF3E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 **result; // rax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rbx
  __int64 *v14; // rax
  char v15; // dl
  __int64 v16; // r15
  __int64 *v17; // rdx
  __int64 v18; // rdi
  __int64 *v19; // rax
  _QWORD *v20; // rax
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rdx
  int v23; // ebx
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 112;
  v8 = *(_QWORD *)(a1 + 96);
  v9 = *(unsigned int *)(a1 + 104);
  while ( 1 )
  {
    result = (__int64 **)(v8 + 24 * v9 - 24);
    v11 = result[1];
    if ( v11 == *result )
      return result;
    v12 = (__int64)(v11 + 1);
    result[1] = v11 + 1;
    v13 = *v11;
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_10;
    v14 = *(__int64 **)(a1 + 8);
    v12 = *(unsigned int *)(a1 + 20);
    v11 = &v14[v12];
    if ( v14 == v11 )
    {
LABEL_15:
      if ( (unsigned int)v12 < *(_DWORD *)(a1 + 16) )
      {
        *(_DWORD *)(a1 + 20) = v12 + 1;
        *v11 = v13;
        LODWORD(v9) = *(_DWORD *)(a1 + 104);
        ++*(_QWORD *)a1;
        goto LABEL_11;
      }
LABEL_10:
      sub_C8CC70(a1, v13, (__int64)v11, v12, a5, a6);
      v9 = *(unsigned int *)(a1 + 104);
      if ( !v15 )
        goto LABEL_9;
LABEL_11:
      v16 = *(_QWORD *)(v13 + 112);
      a6 = v16 + 8LL * *(unsigned int *)(v13 + 120);
      if ( *(_DWORD *)(a1 + 108) <= (unsigned int)v9 )
      {
        v25 = v16 + 8LL * *(unsigned int *)(v13 + 120);
        v8 = sub_C8D7D0(a1 + 96, v6, 0, 0x18u, v26, a6);
        v18 = 3LL * *(unsigned int *)(a1 + 104);
        v19 = (__int64 *)(v18 * 8 + v8);
        if ( v18 * 8 + v8 )
        {
          a6 = v25;
          v19[1] = v16;
          v19[2] = v13;
          *v19 = v25;
          v18 = 3LL * *(unsigned int *)(a1 + 104);
        }
        v20 = *(_QWORD **)(a1 + 96);
        v21 = (unsigned __int64)&v20[v18];
        if ( v20 != (_QWORD *)v21 )
        {
          v22 = (_QWORD *)v8;
          do
          {
            if ( v22 )
            {
              *v22 = *v20;
              v22[1] = v20[1];
              v22[2] = v20[2];
            }
            v20 += 3;
            v22 += 3;
          }
          while ( (_QWORD *)v21 != v20 );
          v21 = *(_QWORD *)(a1 + 96);
        }
        v23 = v26[0];
        if ( v6 != v21 )
          _libc_free(v21);
        v24 = *(_DWORD *)(a1 + 104);
        *(_QWORD *)(a1 + 96) = v8;
        *(_DWORD *)(a1 + 108) = v23;
        v9 = (unsigned int)(v24 + 1);
        *(_DWORD *)(a1 + 104) = v9;
      }
      else
      {
        v8 = *(_QWORD *)(a1 + 96);
        v17 = (__int64 *)(v8 + 24LL * (unsigned int)v9);
        if ( v17 )
        {
          *v17 = a6;
          v17[1] = v16;
          v17[2] = v13;
          LODWORD(v9) = *(_DWORD *)(a1 + 104);
          v8 = *(_QWORD *)(a1 + 96);
        }
        v9 = (unsigned int)(v9 + 1);
        *(_DWORD *)(a1 + 104) = v9;
      }
    }
    else
    {
      while ( v13 != *v14 )
      {
        if ( v11 == ++v14 )
          goto LABEL_15;
      }
      v9 = *(unsigned int *)(a1 + 104);
LABEL_9:
      v8 = *(_QWORD *)(a1 + 96);
    }
  }
}
