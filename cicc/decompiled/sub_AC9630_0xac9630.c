// Function: sub_AC9630
// Address: 0xac9630
//
__int64 __fastcall sub_AC9630(char *src, size_t n, __int64 **a3)
{
  char *v4; // rax
  __int64 v5; // r15
  unsigned int v6; // eax
  unsigned int v7; // r8d
  _QWORD *v8; // r9
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v12; // rax
  unsigned int v13; // r8d
  _QWORD *v14; // r9
  _QWORD *v15; // rcx
  __int64 *v16; // rdx
  __int64 *v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rbx
  _QWORD *v26; // [rsp+8h] [rbp-68h]
  _QWORD *v27; // [rsp+10h] [rbp-60h]
  unsigned int v28; // [rsp+1Ch] [rbp-54h]

  if ( src == &src[n] )
    return sub_AC9350(a3);
  v4 = src;
  while ( !*v4 )
  {
    if ( &src[n] == ++v4 )
      return sub_AC9350(a3);
  }
  v5 = **a3;
  v6 = sub_C92610(src, n);
  v7 = sub_C92740(v5 + 1968, src, n, v6);
  v8 = (_QWORD *)(*(_QWORD *)(v5 + 1968) + 8LL * v7);
  v9 = *v8;
  if ( !*v8 )
    goto LABEL_15;
  if ( v9 == -8 )
  {
    --*(_DWORD *)(v5 + 1984);
LABEL_15:
    v27 = v8;
    v28 = v7;
    v12 = sub_C7D670(n + 17, 8);
    v13 = v28;
    v14 = v27;
    v15 = (_QWORD *)v12;
    if ( n )
    {
      v26 = (_QWORD *)v12;
      memcpy((void *)(v12 + 16), src, n);
      v13 = v28;
      v14 = v27;
      v15 = v26;
    }
    *((_BYTE *)v15 + n + 16) = 0;
    *v15 = n;
    v15[1] = 0;
    *v14 = v15;
    ++*(_DWORD *)(v5 + 1980);
    v16 = (__int64 *)(*(_QWORD *)(v5 + 1968) + 8LL * (unsigned int)sub_C929D0(v5 + 1968, v13));
    v9 = *v16;
    if ( *v16 == -8 || !v9 )
    {
      v17 = v16 + 1;
      do
      {
        do
          v9 = *v17++;
        while ( !v9 );
      }
      while ( v9 == -8 );
    }
  }
  v10 = *(_QWORD *)(v9 + 8);
  if ( v10 )
  {
    while ( *(__int64 ***)(v10 + 8) != a3 )
    {
      if ( !*(_QWORD *)(v10 + 32) )
      {
        v25 = v10 + 32;
        goto LABEL_24;
      }
      v10 = *(_QWORD *)(v10 + 32);
    }
  }
  else
  {
    v25 = v9 + 8;
LABEL_24:
    v18 = v9 + 16;
    if ( *((_BYTE *)a3 + 8) == 16 )
    {
      v24 = sub_BD2C40(40, unk_3F289A4);
      v10 = v24;
      if ( v24 )
      {
        sub_BD35F0(v24, a3, 15);
        *(_QWORD *)(v10 + 24) = v18;
        *(_DWORD *)(v10 + 4) &= 0x38000000u;
        *(_QWORD *)(v10 + 32) = 0;
      }
    }
    else
    {
      v19 = sub_BD2C40(48, unk_3F289A4);
      v10 = v19;
      if ( v19 )
      {
        sub_BD35F0(v19, a3, 16);
        *(_BYTE *)(v10 + 40) &= ~1u;
        *(_DWORD *)(v10 + 4) &= 0x38000000u;
        *(_QWORD *)(v10 + 24) = v18;
        *(_QWORD *)(v10 + 32) = 0;
      }
    }
    v20 = *(_QWORD *)v25;
    *(_QWORD *)v25 = v10;
    if ( v20 )
    {
      v21 = *(_QWORD *)(v20 + 32);
      if ( v21 )
      {
        v22 = *(_QWORD *)(v21 + 32);
        if ( v22 )
        {
          v23 = *(_QWORD *)(v22 + 32);
          if ( v23 )
          {
            sub_AC5B80((__int64 *)(v23 + 32));
            sub_BD7260(v23);
            sub_BD2DD0(v23);
          }
          sub_BD7260(v22);
          sub_BD2DD0(v22);
        }
        sub_BD7260(v21);
        sub_BD2DD0(v21);
      }
      sub_BD7260(v20);
      sub_BD2DD0(v20);
      return *(_QWORD *)v25;
    }
  }
  return v10;
}
