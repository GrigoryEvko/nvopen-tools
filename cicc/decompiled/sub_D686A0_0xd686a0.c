// Function: sub_D686A0
// Address: 0xd686a0
//
_QWORD *__fastcall sub_D686A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  __int64 v8; // r14
  char v9; // r15
  unsigned int v10; // eax
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v15; // rbx
  _BYTE *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-258h]
  _BYTE v26[592]; // [rsp+10h] [rbp-250h] BYREF

  v7 = a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( (unsigned int)a2 <= 4 )
  {
    if ( !v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v15 = a1 + 16;
    v25 = a1 + 560;
  }
  else
  {
    v10 = sub_AF1560((unsigned int)(a2 - 1));
    v7 = v10;
    if ( v10 > 0x40 )
    {
      a4 = a1 + 560;
      v15 = a1 + 16;
      v25 = a1 + 560;
      if ( !v9 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 136LL * v10;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v9 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 8704;
        v7 = 64;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
        sub_D68390(a1, v8, v8 + 136LL * v11);
        return (_QWORD *)sub_C7D6A0(v8, 136LL * v11, 8);
      }
      v15 = a1 + 16;
      v7 = 64;
      v25 = a1 + 560;
    }
  }
  v16 = v26;
  do
  {
    v17 = *(_QWORD *)v15;
    if ( *(_QWORD *)v15 != -4096 && v17 != -8192 )
    {
      if ( v16 )
        *(_QWORD *)v16 = v17;
      v18 = *(_QWORD *)(v15 + 16);
      v19 = *(unsigned int *)(v15 + 48);
      *((_QWORD *)v16 + 6) = 0x200000000LL;
      ++*(_QWORD *)(v15 + 8);
      *((_QWORD *)v16 + 2) = v18;
      LODWORD(v18) = *(_DWORD *)(v15 + 24);
      *((_QWORD *)v16 + 1) = 1;
      *((_DWORD *)v16 + 6) = v18;
      LODWORD(v18) = *(_DWORD *)(v15 + 28);
      *(_QWORD *)(v15 + 16) = 0;
      *((_DWORD *)v16 + 7) = v18;
      LODWORD(v18) = *(_DWORD *)(v15 + 32);
      *(_DWORD *)(v15 + 24) = 0;
      *((_DWORD *)v16 + 8) = v18;
      *(_DWORD *)(v15 + 28) = 0;
      *(_DWORD *)(v15 + 32) = 0;
      *((_QWORD *)v16 + 5) = v16 + 56;
      if ( (_DWORD)v19 )
      {
        a2 = v15 + 40;
        sub_D67C10((__int64)(v16 + 40), (char **)(v15 + 40), v19, a4, a5, a6);
      }
      v20 = *(_QWORD *)(v15 + 80);
      ++*(_QWORD *)(v15 + 72);
      *((_QWORD *)v16 + 9) = 1;
      *((_QWORD *)v16 + 10) = v20;
      LODWORD(v20) = *(_DWORD *)(v15 + 88);
      *(_QWORD *)(v15 + 80) = 0;
      *((_DWORD *)v16 + 22) = v20;
      LODWORD(v20) = *(_DWORD *)(v15 + 92);
      *(_DWORD *)(v15 + 88) = 0;
      *((_DWORD *)v16 + 23) = v20;
      LODWORD(v20) = *(_DWORD *)(v15 + 96);
      *(_DWORD *)(v15 + 92) = 0;
      *((_DWORD *)v16 + 24) = v20;
      *((_QWORD *)v16 + 13) = v16 + 120;
      LODWORD(v20) = *(_DWORD *)(v15 + 112);
      *(_DWORD *)(v15 + 96) = 0;
      *((_QWORD *)v16 + 14) = 0x200000000LL;
      if ( (_DWORD)v20 )
      {
        a2 = v15 + 104;
        sub_D67C10((__int64)(v16 + 104), (char **)(v15 + 104), v19, a4, a5, a6);
      }
      v21 = *(_QWORD *)(v15 + 104);
      v16 += 136;
      if ( v21 != v15 + 120 )
        _libc_free(v21, a2);
      v22 = 8LL * *(unsigned int *)(v15 + 96);
      sub_C7D6A0(*(_QWORD *)(v15 + 80), v22, 8);
      v23 = *(_QWORD *)(v15 + 40);
      if ( v23 != v15 + 56 )
        _libc_free(v23, v22);
      a2 = 8LL * *(unsigned int *)(v15 + 32);
      sub_C7D6A0(*(_QWORD *)(v15 + 16), a2, 8);
    }
    v15 += 136;
  }
  while ( v25 != v15 );
  if ( v7 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v24 = sub_C7D670(136LL * v7, 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v24;
  }
  return sub_D68390(a1, (__int64)v26, (__int64)v16);
}
