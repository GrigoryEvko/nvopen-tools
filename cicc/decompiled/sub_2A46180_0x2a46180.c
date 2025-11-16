// Function: sub_2A46180
// Address: 0x2a46180
//
__int64 __fastcall sub_2A46180(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rax
  _QWORD *v14; // rax
  char v15; // dl
  __int64 v16; // r14
  unsigned __int64 *v17; // r13
  _QWORD *v18; // rdx
  _QWORD *v19; // rsi
  _QWORD *v20; // r13
  _QWORD *v21; // r12
  __int64 v22; // rax
  _QWORD *v23; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // rdi
  char *v27; // r13
  unsigned __int64 *v28; // [rsp+8h] [rbp-38h]

  v28 = a3;
  if ( *(_QWORD *)(a2 + 536) )
  {
    v14 = sub_2A45D40(a2 + 496, a3);
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
        if ( v9 <= 0x13 )
          goto LABEL_21;
        goto LABEL_11;
      }
    }
    if ( v11 != (unsigned __int64 *)v12 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v12;
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
    if ( v9 > 0x13 )
    {
LABEL_11:
      v16 = a2 + 496;
      v17 = (unsigned __int64 *)v8;
      do
      {
        v19 = sub_2A46080((_QWORD *)(a2 + 496), a2 + 504, (__int64)v17);
        if ( v18 )
          sub_2A444D0(a2 + 496, (__int64)v19, v18, v17);
        v17 += 3;
      }
      while ( v11 != v17 );
      v20 = *(_QWORD **)a2;
      v21 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
      while ( v21 != v20 )
      {
        while ( 1 )
        {
          v22 = *(v21 - 1);
          v21 -= 3;
          if ( v22 == 0 || v22 == -4096 || v22 == -8192 )
            break;
          sub_BD60C0(v21);
          if ( v21 == v20 )
            goto LABEL_20;
        }
      }
      goto LABEL_20;
    }
    v24 = v9 + 1;
    if ( *(unsigned int *)(a2 + 12) >= v24 )
      goto LABEL_23;
    goto LABEL_32;
  }
  if ( v9 <= 0x13 )
  {
LABEL_21:
    v24 = v9 + 1;
    if ( v24 <= *(unsigned int *)(a2 + 12) )
    {
LABEL_22:
      if ( v11 )
      {
LABEL_23:
        *v11 = 0;
        v11[1] = 0;
        v25 = v28[2];
        v11[2] = v25;
        if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
          sub_BD6050(v11, *v28 & 0xFFFFFFFFFFFFFFF8LL);
        v8 = *(_QWORD *)a2;
        LODWORD(v10) = *(_DWORD *)(a2 + 8);
      }
      v26 = (unsigned int)(v10 + 1);
      *(_DWORD *)(a2 + 8) = v26;
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v8 + 24 * v26 - 24;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
LABEL_32:
    if ( v8 > (unsigned __int64)v28 || v11 <= v28 )
    {
      sub_2A45F50(a2, v24, (__int64)a3, v8, a5, a6);
      v8 = *(_QWORD *)a2;
      v10 = *(unsigned int *)(a2 + 8);
      v11 = (unsigned __int64 *)(*(_QWORD *)a2 + 24 * v10);
    }
    else
    {
      v27 = (char *)v28 - v8;
      sub_2A45F50(a2, v24, (__int64)a3, v8, a5, a6);
      v8 = *(_QWORD *)a2;
      v28 = (unsigned __int64 *)&v27[*(_QWORD *)a2];
      v10 = *(unsigned int *)(a2 + 8);
      v11 = (unsigned __int64 *)(*(_QWORD *)a2 + 24 * v10);
    }
    goto LABEL_22;
  }
  v16 = a2 + 496;
LABEL_20:
  *(_DWORD *)(a2 + 8) = 0;
  v23 = sub_2A45D40(v16, v28);
  *(_BYTE *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v23;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
