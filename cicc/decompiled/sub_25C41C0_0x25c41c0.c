// Function: sub_25C41C0
// Address: 0x25c41c0
//
void __fastcall sub_25C41C0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v4; // rbx
  bool v5; // zf
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *i; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  int v11; // esi
  int v12; // r9d
  unsigned int v13; // edx
  __int64 *v14; // r8
  __int64 *v15; // r12
  __int64 v16; // rcx
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rsi
  __int64 v21; // r13
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  int v24; // esi
  __int64 v26; // [rsp+8h] [rbp-38h]

  v4 = a2;
  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    v6 = *(_QWORD **)(a1 + 16);
    v7 = 52LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v6 = (_QWORD *)(a1 + 16);
    v7 = 832;
  }
  for ( i = &v6[v7]; i != v6; v6 += 52 )
  {
    if ( v6 )
      *v6 = -4096;
  }
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v9 = *v4;
      if ( *v4 == -8192 || v9 == -4096 )
        goto LABEL_35;
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v10 = a1 + 16;
        v11 = 15;
      }
      else
      {
        v24 = *(_DWORD *)(a1 + 24);
        v10 = *(_QWORD *)(a1 + 16);
        if ( !v24 )
        {
          MEMORY[0] = *v4;
          BUG();
        }
        v11 = v24 - 1;
      }
      v12 = 1;
      v13 = v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v14 = 0;
      v15 = (__int64 *)(v10 + 416LL * v13);
      v16 = *v15;
      if ( v9 != *v15 )
      {
        while ( v16 != -4096 )
        {
          if ( v16 == -8192 && !v14 )
            v14 = v15;
          v13 = v11 & (v12 + v13);
          v15 = (__int64 *)(v10 + 416LL * v13);
          v16 = *v15;
          if ( v9 == *v15 )
            goto LABEL_13;
          ++v12;
        }
        if ( v14 )
          v15 = v14;
      }
LABEL_13:
      *v15 = v9;
      v15[1] = 0;
      v17 = v15 + 3;
      v15[2] = 1;
      do
      {
        if ( v17 )
          *v17 = -4096;
        v17 += 12;
      }
      while ( v17 != v15 + 51 );
      sub_25C3E80((__int64)(v15 + 1), (__int64)(v4 + 1));
      *((_BYTE *)v15 + 408) = *((_BYTE *)v4 + 408);
      *((_BYTE *)v15 + 409) = *((_BYTE *)v4 + 409);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( (v4[2] & 1) != 0 )
        break;
      v18 = *((unsigned int *)v4 + 8);
      v19 = v4[3];
      v20 = 96 * v18;
      if ( (_DWORD)v18 )
      {
        v26 = v19 + v20;
        if ( v19 + v20 != v19 )
          goto LABEL_20;
      }
LABEL_40:
      sub_C7D6A0(v19, v20, 8);
LABEL_35:
      v4 += 52;
      if ( a3 == v4 )
        return;
    }
    v19 = (__int64)(v4 + 3);
    v26 = (__int64)(v4 + 51);
    do
    {
LABEL_20:
      if ( *(_QWORD *)v19 != -8192 && *(_QWORD *)v19 != -4096 )
      {
        v21 = *(_QWORD *)(v19 + 16);
        v22 = v21 + 32LL * *(unsigned int *)(v19 + 24);
        if ( v21 != v22 )
        {
          do
          {
            v22 -= 32LL;
            if ( *(_DWORD *)(v22 + 24) > 0x40u )
            {
              v23 = *(_QWORD *)(v22 + 16);
              if ( v23 )
                j_j___libc_free_0_0(v23);
            }
            if ( *(_DWORD *)(v22 + 8) > 0x40u && *(_QWORD *)v22 )
              j_j___libc_free_0_0(*(_QWORD *)v22);
          }
          while ( v21 != v22 );
          v22 = *(_QWORD *)(v19 + 16);
        }
        if ( v22 != v19 + 32 )
          _libc_free(v22);
      }
      v19 += 96;
    }
    while ( v19 != v26 );
    if ( (v4[2] & 1) != 0 )
      goto LABEL_35;
    v19 = v4[3];
    v20 = 96LL * *((unsigned int *)v4 + 8);
    goto LABEL_40;
  }
}
