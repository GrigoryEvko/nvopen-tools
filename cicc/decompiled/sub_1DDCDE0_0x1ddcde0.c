// Function: sub_1DDCDE0
// Address: 0x1ddcde0
//
__int64 __fastcall sub_1DDCDE0(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // r15
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // r12d
  _QWORD *v11; // rbx
  unsigned int v13; // eax
  unsigned int v14; // eax
  int v15; // edx
  unsigned __int64 v16; // r9
  int v17; // edx
  __int64 v18; // r8
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  int v23; // r10d
  _QWORD *v24; // [rsp+8h] [rbp-B8h]
  __int64 v25; // [rsp+18h] [rbp-A8h]
  unsigned int v26; // [rsp+2Ch] [rbp-94h] BYREF
  unsigned __int64 v27[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v28[64]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v29; // [rsp+80h] [rbp-40h]
  char v30; // [rsp+88h] [rbp-38h]

  v3 = a1;
  v5 = *a3;
  v6 = *(_QWORD *)(a1 + 64);
  v27[0] = (unsigned __int64)v28;
  v27[1] = 0x400000000LL;
  v29 = 0;
  v30 = 0;
  v7 = *(_QWORD *)(v6 + 24 * v5 + 8);
  if ( !v7 || !*(_BYTE *)(v7 + 8) )
  {
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v5);
    v24 = *(_QWORD **)(v25 + 96);
    if ( *(_QWORD **)(v25 + 88) != v24 )
    {
      v11 = *(_QWORD **)(v25 + 88);
      while ( 1 )
      {
        v14 = sub_1DF1770(*(_QWORD *)(a1 + 112), v25, v11);
        v15 = *(_DWORD *)(a1 + 184);
        v16 = v14;
        v13 = -1;
        if ( v15 )
        {
          v17 = v15 - 1;
          v18 = *(_QWORD *)(a1 + 168);
          v19 = v17 & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
          v20 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v20;
          if ( *v11 == *v20 )
          {
LABEL_14:
            v13 = *((_DWORD *)v20 + 2);
          }
          else
          {
            v22 = 1;
            while ( v21 != -8 )
            {
              v23 = v22 + 1;
              v19 = v17 & (v22 + v19);
              v20 = (__int64 *)(v18 + 16LL * v19);
              v21 = *v20;
              if ( *v11 == *v20 )
                goto LABEL_14;
              v22 = v23;
            }
            v13 = -1;
          }
        }
        v26 = v13;
        if ( !sub_13710E0(a1, (__int64)v27, a2, a3, &v26, v16) )
          goto LABEL_11;
        if ( v24 == ++v11 )
        {
          v3 = a1;
          break;
        }
      }
    }
LABEL_7:
    sub_1373530(v3, a3, a2, (__int64)v27);
    v9 = 1;
    goto LABEL_8;
  }
  do
  {
    v8 = v7;
    v7 = *(_QWORD *)v7;
  }
  while ( v7 && *(_BYTE *)(v7 + 8) );
  if ( sub_1371320(a1, a2, v8, (__int64)v27) )
    goto LABEL_7;
LABEL_11:
  v9 = 0;
LABEL_8:
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0]);
  return v9;
}
