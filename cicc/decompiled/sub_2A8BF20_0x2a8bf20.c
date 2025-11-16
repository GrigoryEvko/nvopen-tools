// Function: sub_2A8BF20
// Address: 0x2a8bf20
//
void __fastcall sub_2A8BF20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r15
  unsigned __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  unsigned int v13; // eax
  const void **v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdi
  int v21; // r15d
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // r12
  bool v25; // cc
  __int64 v26; // r12
  __int64 v27; // rbx
  unsigned __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // [rsp-60h] [rbp-60h]
  __int64 v32; // [rsp-58h] [rbp-58h]
  unsigned __int64 v33; // [rsp-58h] [rbp-58h]
  unsigned int v34; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v35; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)a1;
    v34 = *(_DWORD *)(a2 + 8);
    v8 = v34;
    if ( v34 <= v9 )
    {
      v16 = *(_QWORD *)a1;
      if ( v34 )
      {
        v22 = v10 + 8;
        v23 = *(_QWORD *)a2 + 8LL;
        v32 = 24LL * v34;
        v24 = v23 + v32;
        do
        {
          v25 = *(_DWORD *)(v22 + 8) <= 0x40u;
          *(_QWORD *)(v22 - 8) = *(_QWORD *)(v23 - 8);
          if ( v25 && *(_DWORD *)(v23 + 8) <= 0x40u )
          {
            *(_QWORD *)v22 = *(_QWORD *)v23;
            *(_DWORD *)(v22 + 8) = *(_DWORD *)(v23 + 8);
          }
          else
          {
            sub_C43990(v22, v23);
          }
          v23 += 24;
          v22 += 24;
        }
        while ( v24 != v23 );
        v16 = *(_QWORD *)a1;
        v9 = *(unsigned int *)(a1 + 8);
        v10 += v32;
      }
      v17 = v16 + 24 * v9;
      while ( v10 != v17 )
      {
        v17 -= 24LL;
        if ( *(_DWORD *)(v17 + 16) > 0x40u )
        {
          v18 = *(_QWORD *)(v17 + 8);
          if ( v18 )
            j_j___libc_free_0_0(v18);
        }
      }
LABEL_12:
      *(_DWORD *)(a1 + 8) = v34;
      return;
    }
    if ( v34 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v19 = v10 + 24 * v9;
      while ( v19 != v10 )
      {
        while ( 1 )
        {
          v19 -= 24LL;
          if ( *(_DWORD *)(v19 + 16) <= 0x40u )
            break;
          v20 = *(_QWORD *)(v19 + 8);
          if ( !v20 )
            break;
          j_j___libc_free_0_0(v20);
          if ( v19 == v10 )
            goto LABEL_25;
        }
      }
LABEL_25:
      *(_DWORD *)(a1 + 8) = 0;
      v10 = sub_C8D7D0(a1, a1 + 16, v34, 0x18u, &v35, a6);
      sub_2A8AC40((__int64 *)a1, v10);
      v21 = v35;
      if ( a1 + 16 != *(_QWORD *)a1 )
        _libc_free(*(_QWORD *)a1);
      *(_QWORD *)a1 = v10;
      v9 = 0;
      *(_DWORD *)(a1 + 12) = v21;
      v8 = *(unsigned int *)(a2 + 8);
    }
    else if ( *(_DWORD *)(a1 + 8) )
    {
      v26 = v10 + 8;
      v27 = *(_QWORD *)a2 + 8LL;
      v31 = 24 * v9;
      v9 *= 24LL;
      v28 = v27 + v9;
      do
      {
        while ( 1 )
        {
          v25 = *(_DWORD *)(v26 + 8) <= 0x40u;
          *(_QWORD *)(v26 - 8) = *(_QWORD *)(v27 - 8);
          if ( v25 && *(_DWORD *)(v27 + 8) <= 0x40u )
            break;
          v29 = v27;
          v27 += 24;
          v33 = v9;
          sub_C43990(v26, v29);
          v26 += 24;
          v9 = v33;
          if ( v28 == v27 )
            goto LABEL_40;
        }
        v30 = *(_QWORD *)v27;
        v27 += 24;
        v26 += 24;
        *(_QWORD *)(v26 - 24) = v30;
        *(_DWORD *)(v26 - 16) = *(_DWORD *)(v27 - 16);
      }
      while ( v28 != v27 );
LABEL_40:
      v8 = *(unsigned int *)(a2 + 8);
      v10 = *(_QWORD *)a1 + v31;
    }
    v11 = *(_QWORD *)a2 + 24 * v8;
    v12 = v9 + *(_QWORD *)a2;
    if ( v11 == v12 )
      goto LABEL_12;
    while ( 1 )
    {
      if ( !v10 )
        goto LABEL_8;
      *(_QWORD *)v10 = *(_QWORD *)v12;
      v13 = *(_DWORD *)(v12 + 16);
      *(_DWORD *)(v10 + 16) = v13;
      if ( v13 <= 0x40 )
      {
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(v12 + 8);
LABEL_8:
        v12 += 24LL;
        v10 += 24;
        if ( v11 == v12 )
          goto LABEL_12;
      }
      else
      {
        v14 = (const void **)(v12 + 8);
        v15 = v10 + 8;
        v12 += 24LL;
        v10 += 24;
        sub_C43780(v15, v14);
        if ( v11 == v12 )
          goto LABEL_12;
      }
    }
  }
}
