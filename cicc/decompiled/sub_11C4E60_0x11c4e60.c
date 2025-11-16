// Function: sub_11C4E60
// Address: 0x11c4e60
//
__int64 __fastcall sub_11C4E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v10; // r12
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v20; // r12
  unsigned __int8 *v21; // rdi
  __int64 v22; // [rsp+0h] [rbp-40h]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  v10 = v6;
  if ( !(_DWORD)v7 )
    goto LABEL_29;
  v11 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v14 )
  {
    v13 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F81450 && a3 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_29;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v8 + 24 * v7 )
  {
LABEL_29:
    v15 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    if ( v15 )
      v15 += 8;
  }
  v16 = a3 + 72;
  v17 = *(_QWORD *)(a3 + 80);
  if ( v16 != v17 )
  {
    if ( !v17 )
      BUG();
    while ( 1 )
    {
      v18 = *(_QWORD *)(v17 + 32);
      if ( v18 != v17 + 24 )
        break;
      v17 = *(_QWORD *)(v17 + 8);
      if ( v16 == v17 )
        goto LABEL_16;
      if ( !v17 )
        BUG();
    }
    if ( v16 != v17 )
    {
      v20 = v10 + 8;
      do
      {
        v21 = (unsigned __int8 *)(v18 - 24);
        v22 = a1;
        if ( !v18 )
          v21 = 0;
        v23 = v15;
        sub_11C4E30(v21, v20, v15);
        v18 = *(_QWORD *)(v18 + 8);
        a1 = v22;
        v15 = v23;
        while ( v18 == v17 - 24 + 48 )
        {
          v17 = *(_QWORD *)(v17 + 8);
          if ( v16 == v17 )
            goto LABEL_16;
          if ( !v17 )
            BUG();
          v18 = *(_QWORD *)(v17 + 32);
        }
      }
      while ( v16 != v17 );
    }
  }
LABEL_16:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82408;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
