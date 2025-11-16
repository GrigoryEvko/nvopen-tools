// Function: sub_2E1B600
// Address: 0x2e1b600
//
void __fastcall sub_2E1B600(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rsi
  _DWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rsi
  unsigned int v16; // edi
  _QWORD *v17; // rcx
  int v18; // r9d
  unsigned int v19; // edx
  __int64 v20; // r8
  __int64 v21; // rdi
  _QWORD *v22; // rsi
  _QWORD *v23; // rcx
  int v24; // r9d
  __int64 v25; // rdx
  unsigned __int64 i; // r8
  __int64 v27; // r10
  __int64 v28; // rsi
  __int64 v29; // r9
  unsigned int v30; // edx
  __int64 v31; // rdi
  __int64 *v32; // [rsp-A8h] [rbp-A8h]
  unsigned int v33; // [rsp-9Ch] [rbp-9Ch]
  _DWORD *v34; // [rsp-98h] [rbp-98h] BYREF
  _QWORD *v35; // [rsp-90h] [rbp-90h]
  __int64 v36; // [rsp-88h] [rbp-88h]
  _QWORD v37[16]; // [rsp-80h] [rbp-80h] BYREF

  if ( *(_DWORD *)(a3 + 8) )
  {
    v5 = (__int64)a1;
    v6 = a3;
    ++*a1;
    v7 = *(__int64 **)a3;
    v8 = (unsigned int)a1[50];
    v9 = *(_QWORD *)a3 + 24LL * *(unsigned int *)(a3 + 8);
    v10 = **(_QWORD **)a3;
    v35 = v37;
    v32 = (__int64 *)v9;
    v11 = a1 + 2;
    v34 = a1 + 2;
    v36 = 0x400000000LL;
    if ( (_DWORD)v8 )
    {
      sub_2E1A860((__int64)&v34, v10, a3, (__int64)a1, a5, v8);
      v11 = v34;
      LODWORD(v8) = v34[48];
    }
    else
    {
      v12 = (unsigned int)a1[51];
      if ( (_DWORD)v12 )
      {
        v13 = v10;
        v14 = v10;
        v5 += 16;
        v15 = 0;
        v16 = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v14 >> 1) & 3;
        do
        {
          a3 = *(_DWORD *)((*(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v5 >> 1) & 3;
          if ( (unsigned int)a3 > v16 )
            break;
          v15 = (unsigned int)(v15 + 1);
          v5 += 16;
        }
        while ( (_DWORD)v12 != (_DWORD)v15 );
      }
      else
      {
        v15 = 0;
      }
      v37[0] = v11;
      LODWORD(v36) = 1;
      v37[1] = v12 | (v15 << 32);
    }
    if ( (_DWORD)v8 )
      goto LABEL_25;
LABEL_9:
    v17 = v35;
    v18 = v11[49];
    v19 = HIDWORD(v35[2 * (unsigned int)v36 - 1]) + 1;
    if ( v18 != v19 )
    {
      do
      {
        v20 = v19;
        v21 = v19++ - 1;
        v22 = &v11[4 * v20];
        v23 = &v11[4 * v21];
        *v23 = *v22;
        v23[1] = v22[1];
        *(_QWORD *)&v11[2 * v21 + 32] = *(_QWORD *)&v11[2 * v20 + 32];
      }
      while ( v18 != v19 );
      v17 = v35;
    }
    v24 = v18 - 1;
    v11[49] = v24;
    *((_DWORD *)v17 + 2) = v24;
    v25 = (unsigned int)v36;
    for ( i = (unsigned __int64)v35; (_DWORD)v36; i = (unsigned __int64)v35 )
    {
      if ( *(_DWORD *)(i + 12) >= *(_DWORD *)(i + 8) )
        break;
      v11 = v34;
      v27 = i + 16 * v25 - 16;
      v28 = *(unsigned int *)(v27 + 12);
      v33 = v34[48];
      v5 = v28;
      v29 = 24LL * *(unsigned int *)(v6 + 8);
      v30 = *(_DWORD *)((*(_QWORD *)(*(_QWORD *)v27 + 16 * v28) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (*(__int64 *)(*(_QWORD *)v27 + 16 * v28) >> 1) & 3;
      if ( v30 < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v6 + v29 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                | (unsigned int)(*(__int64 *)(*(_QWORD *)v6 + v29 - 16) >> 1) & 3) )
      {
        if ( (*(_DWORD *)((v7[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7[1] >> 1) & 3) <= v30 )
        {
          do
          {
            v31 = v7[4];
            v7 += 3;
          }
          while ( v30 >= (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v31 >> 1) & 3) );
        }
      }
      else
      {
        v7 = (__int64 *)(*(_QWORD *)v6 + v29);
      }
      if ( v32 == v7 )
        break;
      a3 = v33;
      if ( v33 )
      {
        sub_2E1A970((__int64)&v34, *v7);
        v11 = v34;
      }
      else
      {
        v12 = (unsigned int)v34[49];
        if ( (_DWORD)v12 != (_DWORD)v28 )
        {
          while ( 1 )
          {
            a3 = *(_DWORD *)((*(_QWORD *)&v34[4 * v28 + 2] & 0xFFFFFFFFFFFFFFF8LL) + 24)
               | (unsigned int)(*(__int64 *)&v34[4 * v28 + 2] >> 1) & 3;
            if ( (unsigned int)a3 > (*(_DWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v7 >> 1) & 3) )
              break;
            v5 = (unsigned int)(v5 + 1);
            if ( (_DWORD)v12 == (_DWORD)v5 )
              break;
            v28 = (unsigned int)v5;
          }
        }
        *(_DWORD *)(v27 + 12) = v5;
      }
      if ( !v11[48] )
        goto LABEL_9;
LABEL_25:
      sub_2E1B3E0((__int64)&v34, 1, a3, v5, v12);
      v25 = (unsigned int)v36;
    }
    if ( (_QWORD *)i != v37 )
      _libc_free(i);
  }
}
