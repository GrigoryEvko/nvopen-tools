// Function: sub_D6C8F0
// Address: 0xd6c8f0
//
__int64 __fastcall sub_D6C8F0(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rax
  _QWORD *v14; // rax
  char v15; // dl
  unsigned __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r14
  unsigned __int64 *v19; // r15
  __int64 v20; // rdx
  _QWORD *v21; // rsi
  _QWORD *v22; // r12
  _QWORD *v23; // r15
  _QWORD *v24; // rax
  char *v25; // r12
  unsigned __int64 *v26; // [rsp+8h] [rbp-38h]

  v26 = a3;
  if ( *(_QWORD *)(a2 + 248) )
  {
    v14 = sub_D6B980(a2 + 208, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v14;
    *(_BYTE *)(a1 + 16) = v15;
    return a1;
  }
  v8 = *(_QWORD *)a2;
  v9 = *(unsigned int *)(a2 + 8);
  LODWORD(v10) = v9;
  v11 = (unsigned __int64 *)(v8 + 24 * v9);
  if ( (unsigned __int64 *)v8 != v11 )
  {
    a3 = (unsigned __int64 *)a3[2];
    v12 = v8;
    while ( *(unsigned __int64 **)(v12 + 16) != a3 )
    {
      v12 += 24LL;
      if ( v11 == (unsigned __int64 *)v12 )
      {
        if ( v9 <= 7 )
          goto LABEL_11;
        goto LABEL_15;
      }
    }
    if ( v11 != (unsigned __int64 *)v12 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v12;
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
    if ( v9 > 7 )
    {
LABEL_15:
      v18 = a2 + 208;
      v19 = (unsigned __int64 *)v8;
      do
      {
        v21 = sub_D6C7F0((_QWORD *)(a2 + 208), (_QWORD *)(a2 + 216), (__int64)v19);
        if ( v20 )
          sub_D68100(a2 + 208, (__int64)v21, v20, v19);
        v19 += 3;
      }
      while ( v11 != v19 );
      v22 = *(_QWORD **)a2;
      v23 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
      if ( v23 != *(_QWORD **)a2 )
      {
        do
        {
          v23 -= 3;
          sub_D68D70(v23);
        }
        while ( v22 != v23 );
      }
      goto LABEL_21;
    }
    v16 = v9 + 1;
    if ( *(unsigned int *)(a2 + 12) >= v16 )
      goto LABEL_13;
    goto LABEL_24;
  }
  if ( v9 <= 7 )
  {
LABEL_11:
    v16 = v9 + 1;
    if ( v16 <= *(unsigned int *)(a2 + 12) )
    {
LABEL_12:
      if ( v11 )
      {
LABEL_13:
        sub_D68CD0(v11, 0, v26);
        v8 = *(_QWORD *)a2;
        LODWORD(v10) = *(_DWORD *)(a2 + 8);
      }
      v17 = (unsigned int)(v10 + 1);
      *(_DWORD *)(a2 + 8) = v17;
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v8 + 24 * v17 - 24;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
LABEL_24:
    if ( v8 > (unsigned __int64)v26 || v11 <= v26 )
    {
      sub_D6B530(a2, v16, (__int64)a3, v8, a5, a6);
      v8 = *(_QWORD *)a2;
      v10 = *(unsigned int *)(a2 + 8);
      v11 = (unsigned __int64 *)(*(_QWORD *)a2 + 24 * v10);
    }
    else
    {
      v25 = (char *)v26 - v8;
      sub_D6B530(a2, v16, (__int64)a3, v8, a5, a6);
      v8 = *(_QWORD *)a2;
      v26 = (unsigned __int64 *)&v25[*(_QWORD *)a2];
      v10 = *(unsigned int *)(a2 + 8);
      v11 = (unsigned __int64 *)(*(_QWORD *)a2 + 24 * v10);
    }
    goto LABEL_12;
  }
  v18 = a2 + 208;
LABEL_21:
  *(_DWORD *)(a2 + 8) = 0;
  v24 = sub_D6B980(v18, v26);
  *(_BYTE *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v24;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
