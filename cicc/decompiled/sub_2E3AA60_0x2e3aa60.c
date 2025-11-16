// Function: sub_2E3AA60
// Address: 0x2e3aa60
//
__int64 __fastcall sub_2E3AA60(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // r15
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r12d
  __int64 v12; // rax
  __int64 *v13; // rbx
  int v15; // ecx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r8
  unsigned int v19; // eax
  unsigned int v20; // eax
  int v21; // ecx
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // r9
  int v25; // eax
  int v26; // r10d
  __int64 *v27; // [rsp+8h] [rbp-B8h]
  __int64 v28; // [rsp+18h] [rbp-A8h]
  unsigned int v29; // [rsp+2Ch] [rbp-94h] BYREF
  unsigned __int64 v30[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v31[64]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v32; // [rsp+80h] [rbp-40h]
  char v33; // [rsp+88h] [rbp-38h]

  v3 = a1;
  v6 = *a3;
  v7 = *(_QWORD *)(a1 + 64);
  v30[0] = (unsigned __int64)v31;
  v30[1] = 0x400000000LL;
  v32 = 0;
  v33 = 0;
  v8 = *(_QWORD *)(v7 + 24 * v6 + 8);
  if ( !v8 || !*(_BYTE *)(v8 + 8) )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v6);
    v13 = *(__int64 **)(v12 + 112);
    v28 = v12;
    v27 = &v13[*(unsigned int *)(v12 + 120)];
    if ( v27 != v13 )
    {
      while ( 1 )
      {
        v20 = sub_2E441C0(*(_QWORD *)(a1 + 112), v28, v13);
        v21 = *(_DWORD *)(a1 + 184);
        v22 = *v13;
        v23 = *(_QWORD *)(a1 + 168);
        v24 = v20;
        if ( !v21 )
          goto LABEL_19;
        v15 = v21 - 1;
        v16 = v15 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v17 = (__int64 *)(v23 + 16LL * v16);
        v18 = *v17;
        if ( v22 != *v17 )
          break;
LABEL_15:
        v19 = *((_DWORD *)v17 + 2);
LABEL_16:
        v29 = v19;
        if ( !sub_FE8BD0(a1, (__int64)v30, a2, a3, &v29, v24) )
          goto LABEL_11;
        if ( ++v13 == v27 )
        {
          v3 = a1;
          goto LABEL_7;
        }
      }
      v25 = 1;
      while ( v18 != -4096 )
      {
        v26 = v25 + 1;
        v16 = v15 & (v25 + v16);
        v17 = (__int64 *)(v23 + 16LL * v16);
        v18 = *v17;
        if ( v22 == *v17 )
          goto LABEL_15;
        v25 = v26;
      }
LABEL_19:
      v19 = -1;
      goto LABEL_16;
    }
    goto LABEL_7;
  }
  do
  {
    v9 = v8;
    v8 = *(_QWORD *)v8;
  }
  while ( v8 && *(_BYTE *)(v8 + 8) );
  if ( sub_FE8E10(a1, a2, v9, (__int64)v30) )
  {
LABEL_7:
    sub_FEA740(v3, a3, a2, (__int64)v30);
    v10 = 1;
    goto LABEL_8;
  }
LABEL_11:
  v10 = 0;
LABEL_8:
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
  return v10;
}
