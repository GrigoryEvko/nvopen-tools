// Function: sub_2B53D30
// Address: 0x2b53d30
//
__int64 __fastcall sub_2B53D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r15d
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rbx
  __int64 *v11; // rsi
  unsigned __int64 *v12; // r14
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  int v19; // eax
  _QWORD *v20; // rdi
  __int64 v21; // rbx
  char *v23; // r14
  unsigned __int64 v25[2]; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE v26[16]; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v27; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-78h]
  char v29; // [rsp+40h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(a3 + 8);
  sub_2B53530((__int64 *)&v27, a2, a3, a4, a5, a6);
  v8 = v28;
  v9 = v27;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  v10 = v9 + (v8 << 6);
  if ( v10 == v9 )
    goto LABEL_18;
  do
  {
    while ( 1 )
    {
      v11 = *(__int64 **)v9;
      if ( v7 == *(_DWORD *)(v9 + 8) )
        break;
LABEL_3:
      v9 += 64LL;
      if ( v10 == v9 )
        goto LABEL_13;
    }
    v12 = v25;
    v25[0] = (unsigned __int64)v26;
    v25[1] = 0x400000000LL;
    if ( (unsigned __int8)sub_2B3ACB0(a2, v11, v7, (__int64)v25) )
    {
      v15 = *(unsigned int *)(a1 + 8);
      v16 = *(unsigned int *)(a1 + 12);
      v17 = *(_QWORD *)a1;
      v18 = v15 + 1;
      v19 = *(_DWORD *)(a1 + 8);
      if ( v15 + 1 > v16 )
      {
        if ( v17 > (unsigned __int64)v25 || (unsigned __int64)v25 >= v17 + 32 * v15 )
        {
          sub_2B424B0(a1, v18, v17, v16, v13, v14);
          v15 = *(unsigned int *)(a1 + 8);
          v17 = *(_QWORD *)a1;
          v19 = *(_DWORD *)(a1 + 8);
        }
        else
        {
          v23 = (char *)v25 - v17;
          sub_2B424B0(a1, v18, v17, v16, v13, v14);
          v17 = *(_QWORD *)a1;
          v15 = *(unsigned int *)(a1 + 8);
          v12 = (unsigned __int64 *)&v23[*(_QWORD *)a1];
          v19 = *(_DWORD *)(a1 + 8);
        }
      }
      v20 = (_QWORD *)(v17 + 32 * v15);
      if ( !v20 )
        goto LABEL_10;
      *v20 = v20 + 2;
      v20[1] = 0x400000000LL;
      if ( !*((_DWORD *)v12 + 2) )
      {
        v19 = *(_DWORD *)(a1 + 8);
LABEL_10:
        *(_DWORD *)(a1 + 8) = v19 + 1;
        goto LABEL_11;
      }
      sub_2B0D430((__int64)v20, (__int64)v12, v17, v16, v13, v14);
      ++*(_DWORD *)(a1 + 8);
    }
LABEL_11:
    if ( (_BYTE *)v25[0] == v26 )
      goto LABEL_3;
    _libc_free(v25[0]);
    v9 += 64LL;
  }
  while ( v10 != v9 );
LABEL_13:
  v21 = v27;
  v9 = v27 + ((unsigned __int64)v28 << 6);
  if ( v27 != v9 )
  {
    do
    {
      v9 -= 64LL;
      if ( *(_QWORD *)v9 != v9 + 16 )
        _libc_free(*(_QWORD *)v9);
    }
    while ( v9 != v21 );
    v9 = v27;
  }
LABEL_18:
  if ( (char *)v9 != &v29 )
    _libc_free(v9);
  return a1;
}
