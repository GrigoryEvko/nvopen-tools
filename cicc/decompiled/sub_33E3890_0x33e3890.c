// Function: sub_33E3890
// Address: 0x33e3890
//
void __fastcall sub_33E3890(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  unsigned int v17; // edx
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  unsigned int v20; // ecx
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // r14
  int v23; // eax
  __int64 v24; // [rsp+0h] [rbp-50h]
  unsigned __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  int v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( *(unsigned int *)(a1 + 12) < a2 )
  {
    v18 = a2;
    v19 = sub_C8D7D0(a1, a1 + 16, a2, 0x10u, v28, a6);
    v24 = v19;
    while ( 1 )
    {
      if ( !v19 )
        goto LABEL_28;
      v20 = *(_DWORD *)(a3 + 8);
      *(_DWORD *)(v19 + 8) = v20;
      if ( v20 <= 0x40 )
      {
        *(_QWORD *)v19 = *(_QWORD *)a3;
LABEL_28:
        v19 += 16;
        if ( !--v18 )
          goto LABEL_32;
      }
      else
      {
        v26 = v19;
        sub_C43780(v19, (const void **)a3);
        v19 = v26 + 16;
        if ( !--v18 )
        {
LABEL_32:
          v21 = *(_QWORD *)a1;
          v22 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
          if ( *(_QWORD *)a1 != v22 )
          {
            do
            {
              v22 -= 16LL;
              if ( *(_DWORD *)(v22 + 8) > 0x40u && *(_QWORD *)v22 )
                j_j___libc_free_0_0(*(_QWORD *)v22);
            }
            while ( v21 != v22 );
            v22 = *(_QWORD *)a1;
          }
          v23 = v28[0];
          if ( a1 + 16 != v22 )
          {
            v27 = v28[0];
            _libc_free(v22);
            v23 = v27;
          }
          *(_DWORD *)(a1 + 8) = a2;
          *(_DWORD *)(a1 + 12) = v23;
          *(_QWORD *)a1 = v24;
          return;
        }
      }
    }
  }
  v8 = *(unsigned int *)(a1 + 8);
  v9 = a2;
  if ( v8 <= a2 )
    v9 = *(unsigned int *)(a1 + 8);
  if ( v9 )
  {
    v10 = *(_QWORD *)a1;
    v11 = *(_QWORD *)a1 + 16 * v9;
    do
    {
      while ( *(_DWORD *)(v10 + 8) > 0x40u || *(_DWORD *)(a3 + 8) > 0x40u )
      {
        v12 = v10;
        v10 += 16;
        sub_C43990(v12, a3);
        if ( v11 == v10 )
          goto LABEL_10;
      }
      v10 += 16;
      *(_QWORD *)(v10 - 16) = *(_QWORD *)a3;
      *(_DWORD *)(v10 - 8) = *(_DWORD *)(a3 + 8);
    }
    while ( v11 != v10 );
LABEL_10:
    v8 = *(unsigned int *)(a1 + 8);
  }
  if ( v8 < a2 )
  {
    v15 = *(_QWORD *)a1 + 16 * v8;
    v16 = a2 - v8;
    if ( a2 != v8 )
    {
      do
      {
        if ( v15 )
        {
          v17 = *(_DWORD *)(a3 + 8);
          *(_DWORD *)(v15 + 8) = v17;
          if ( v17 <= 0x40 )
          {
            *(_QWORD *)v15 = *(_QWORD *)a3;
          }
          else
          {
            v25 = v16;
            sub_C43780(v15, (const void **)a3);
            v16 = v25;
          }
        }
        v15 += 16;
        --v16;
      }
      while ( v16 );
    }
  }
  else if ( v8 > a2 )
  {
    v13 = *(_QWORD *)a1 + 16 * v8;
    v14 = *(_QWORD *)a1 + 16 * a2;
    while ( v14 != v13 )
    {
      while ( 1 )
      {
        v13 -= 16;
        if ( *(_DWORD *)(v13 + 8) <= 0x40u || !*(_QWORD *)v13 )
          break;
        j_j___libc_free_0_0(*(_QWORD *)v13);
        if ( v14 == v13 )
          goto LABEL_18;
      }
    }
  }
LABEL_18:
  *(_DWORD *)(a1 + 8) = a2;
}
