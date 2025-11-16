// Function: sub_2560D30
// Address: 0x2560d30
//
void __fastcall sub_2560D30(unsigned int *a1, __int64 a2)
{
  unsigned __int64 v4; // rsi
  __int64 v5; // r15
  unsigned __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r15
  unsigned __int64 v10; // r12
  unsigned int v11; // eax
  const void **v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  __int64 v16; // r12
  __int64 v17; // rbx
  unsigned __int64 v18; // r12
  __int64 v19; // r12
  unsigned __int64 v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // [rsp-48h] [rbp-48h]
  unsigned int v24; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 != (unsigned int *)a2 )
  {
    v4 = *(unsigned int *)(a2 + 8);
    v5 = *(_QWORD *)a1;
    v6 = a1[2];
    v24 = v4;
    v7 = *(_QWORD *)a1;
    if ( v4 <= v6 )
    {
      v14 = *(_QWORD *)a1;
      if ( v4 )
      {
        v17 = *(_QWORD *)a2;
        v18 = v5 + 16 * v4;
        do
        {
          if ( *(_DWORD *)(v5 + 8) > 0x40u || *(_DWORD *)(v17 + 8) > 0x40u )
          {
            sub_C43990(v5, v17);
          }
          else
          {
            *(_QWORD *)v5 = *(_QWORD *)v17;
            *(_DWORD *)(v5 + 8) = *(_DWORD *)(v17 + 8);
          }
          v5 += 16;
          v17 += 16;
        }
        while ( v5 != v18 );
        v14 = *(_QWORD *)a1;
        v6 = a1[2];
      }
      v15 = v14 + 16 * v6;
      while ( v5 != v15 )
      {
        v15 -= 16LL;
        if ( *(_DWORD *)(v15 + 8) > 0x40u && *(_QWORD *)v15 )
          j_j___libc_free_0_0(*(_QWORD *)v15);
      }
LABEL_12:
      a1[2] = v24;
      return;
    }
    if ( v4 > a1[3] )
    {
      v16 = v5 + 16 * v6;
      while ( v16 != v5 )
      {
        while ( 1 )
        {
          v16 -= 16;
          if ( *(_DWORD *)(v16 + 8) <= 0x40u || !*(_QWORD *)v16 )
            break;
          j_j___libc_free_0_0(*(_QWORD *)v16);
          if ( v16 == v5 )
            goto LABEL_25;
        }
      }
LABEL_25:
      a1[2] = 0;
      sub_AE4800(a1, v4);
      v4 = *(unsigned int *)(a2 + 8);
      v7 = *(_QWORD *)a1;
      v6 = 0;
    }
    else if ( a1[2] )
    {
      v6 *= 16LL;
      v19 = *(_QWORD *)a2;
      v20 = v5 + v6;
      do
      {
        while ( *(_DWORD *)(v5 + 8) > 0x40u || *(_DWORD *)(v19 + 8) > 0x40u )
        {
          v21 = v5;
          v5 += 16;
          v23 = v6;
          sub_C43990(v21, v19);
          v19 += 16;
          v6 = v23;
          if ( v5 == v20 )
            goto LABEL_38;
        }
        v22 = *(_QWORD *)v19;
        v5 += 16;
        v19 += 16;
        *(_QWORD *)(v5 - 16) = v22;
        *(_DWORD *)(v5 - 8) = *(_DWORD *)(v19 - 8);
      }
      while ( v5 != v20 );
LABEL_38:
      v4 = *(unsigned int *)(a2 + 8);
      v7 = *(_QWORD *)a1;
    }
    v8 = v6 + v7;
    v9 = *(_QWORD *)a2 + 16 * v4;
    v10 = v6 + *(_QWORD *)a2;
    if ( v9 == v10 )
      goto LABEL_12;
    while ( 1 )
    {
      if ( !v8 )
        goto LABEL_8;
      v11 = *(_DWORD *)(v10 + 8);
      *(_DWORD *)(v8 + 8) = v11;
      if ( v11 <= 0x40 )
      {
        *(_QWORD *)v8 = *(_QWORD *)v10;
LABEL_8:
        v10 += 16LL;
        v8 += 16;
        if ( v9 == v10 )
          goto LABEL_12;
      }
      else
      {
        v12 = (const void **)v10;
        v13 = v8;
        v10 += 16LL;
        v8 += 16;
        sub_C43780(v13, v12);
        if ( v9 == v10 )
          goto LABEL_12;
      }
    }
  }
}
