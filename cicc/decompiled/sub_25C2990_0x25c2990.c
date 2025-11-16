// Function: sub_25C2990
// Address: 0x25c2990
//
void __fastcall sub_25C2990(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rsi
  __int64 v5; // r15
  unsigned __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r12
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // eax
  const void **v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // r15
  unsigned __int64 v21; // rdi
  __int64 v22; // rbx
  unsigned __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rcx
  unsigned __int64 v26; // [rsp-48h] [rbp-48h]
  int v27; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 != a2 )
  {
    v4 = *(unsigned int *)(a2 + 8);
    v5 = *(_QWORD *)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v27 = v4;
    v7 = *(_QWORD *)a1;
    if ( v4 <= v6 )
    {
      v17 = *(_QWORD *)a1;
      if ( v4 )
      {
        v22 = *(_QWORD *)a2;
        v23 = v5 + 32 * v4;
        do
        {
          if ( *(_DWORD *)(v5 + 8) > 0x40u || *(_DWORD *)(v22 + 8) > 0x40u )
          {
            sub_C43990(v5, v22);
          }
          else
          {
            *(_QWORD *)v5 = *(_QWORD *)v22;
            *(_DWORD *)(v5 + 8) = *(_DWORD *)(v22 + 8);
          }
          if ( *(_DWORD *)(v5 + 24) <= 0x40u && *(_DWORD *)(v22 + 24) <= 0x40u )
          {
            *(_QWORD *)(v5 + 16) = *(_QWORD *)(v22 + 16);
            *(_DWORD *)(v5 + 24) = *(_DWORD *)(v22 + 24);
          }
          else
          {
            sub_C43990(v5 + 16, v22 + 16);
          }
          v5 += 32;
          v22 += 32;
        }
        while ( v5 != v23 );
        v17 = *(_QWORD *)a1;
        v6 = *(unsigned int *)(a1 + 8);
      }
      v18 = v17 + 32 * v6;
      while ( v5 != v18 )
      {
        v18 -= 32LL;
        if ( *(_DWORD *)(v18 + 24) > 0x40u )
        {
          v19 = *(_QWORD *)(v18 + 16);
          if ( v19 )
            j_j___libc_free_0_0(v19);
        }
        if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
          j_j___libc_free_0_0(*(_QWORD *)v18);
      }
LABEL_14:
      *(_DWORD *)(a1 + 8) = v27;
      return;
    }
    if ( v4 > *(unsigned int *)(a1 + 12) )
    {
      v20 = 32 * v6 + v5;
      while ( v20 != v7 )
      {
        while ( 1 )
        {
          v20 -= 32;
          if ( *(_DWORD *)(v20 + 24) > 0x40u )
          {
            v21 = *(_QWORD *)(v20 + 16);
            if ( v21 )
              j_j___libc_free_0_0(v21);
          }
          if ( *(_DWORD *)(v20 + 8) <= 0x40u || !*(_QWORD *)v20 )
            break;
          j_j___libc_free_0_0(*(_QWORD *)v20);
          if ( v20 == v7 )
            goto LABEL_33;
        }
      }
LABEL_33:
      *(_DWORD *)(a1 + 8) = 0;
      v6 = 0;
      sub_9D5330(a1, v4);
      v4 = *(unsigned int *)(a2 + 8);
      v7 = *(_QWORD *)a1;
    }
    else if ( *(_DWORD *)(a1 + 8) )
    {
      v6 *= 32LL;
      v24 = *(_QWORD *)a2;
      v26 = v5 + v6;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)(v5 + 8) > 0x40u || *(_DWORD *)(v24 + 8) > 0x40u )
          {
            sub_C43990(v5, v24);
          }
          else
          {
            *(_QWORD *)v5 = *(_QWORD *)v24;
            *(_DWORD *)(v5 + 8) = *(_DWORD *)(v24 + 8);
          }
          if ( *(_DWORD *)(v5 + 24) <= 0x40u && *(_DWORD *)(v24 + 24) <= 0x40u )
            break;
          sub_C43990(v5 + 16, v24 + 16);
          v24 += 32;
          v5 += 32;
          if ( v5 == v26 )
            goto LABEL_54;
        }
        v25 = *(_QWORD *)(v24 + 16);
        v5 += 32;
        v24 += 32;
        *(_QWORD *)(v5 - 16) = v25;
        *(_DWORD *)(v5 - 8) = *(_DWORD *)(v24 - 8);
      }
      while ( v5 != v26 );
LABEL_54:
      v4 = *(unsigned int *)(a2 + 8);
      v7 = *(_QWORD *)a1;
    }
    v8 = *(_QWORD *)a2;
    v9 = v6 + v7;
    v10 = v8 + 32 * v4;
    v11 = v6 + v8;
    if ( v10 == v11 )
      goto LABEL_14;
    while ( 1 )
    {
      if ( !v9 )
        goto LABEL_9;
      v13 = *(_DWORD *)(v11 + 8);
      *(_DWORD *)(v9 + 8) = v13;
      if ( v13 > 0x40 )
        break;
      *(_QWORD *)v9 = *(_QWORD *)v11;
      v12 = *(_DWORD *)(v11 + 24);
      *(_DWORD *)(v9 + 24) = v12;
      if ( v12 > 0x40 )
      {
LABEL_13:
        v15 = (const void **)(v11 + 16);
        v16 = v9 + 16;
        v11 += 32LL;
        v9 += 32;
        sub_C43780(v16, v15);
        if ( v10 == v11 )
          goto LABEL_14;
      }
      else
      {
LABEL_8:
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v11 + 16);
LABEL_9:
        v11 += 32LL;
        v9 += 32;
        if ( v10 == v11 )
          goto LABEL_14;
      }
    }
    sub_C43780(v9, (const void **)v11);
    v14 = *(_DWORD *)(v11 + 24);
    *(_DWORD *)(v9 + 24) = v14;
    if ( v14 > 0x40 )
      goto LABEL_13;
    goto LABEL_8;
  }
}
