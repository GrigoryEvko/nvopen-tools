// Function: sub_AE6BA0
// Address: 0xae6ba0
//
__int64 __fastcall sub_AE6BA0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 result; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  unsigned int v11; // r15d
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp-78h] [rbp-78h]
  char v22; // [rsp-70h] [rbp-70h]
  __int64 v23; // [rsp-70h] [rbp-70h]
  __int64 v24; // [rsp-70h] [rbp-70h]
  _BYTE *v25; // [rsp-68h] [rbp-68h] BYREF
  __int64 v26; // [rsp-60h] [rbp-60h]
  _BYTE v27[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_BYTE *)a3 == 6 )
    return 0;
  v4 = a2;
  if ( !*(_BYTE *)(a1 + 28) )
  {
    a2 = (_BYTE *)a3;
    if ( !sub_C8CA60(a1, a3, a3, a4) )
      goto LABEL_10;
    return 0;
  }
  v6 = *(_QWORD **)(a1 + 8);
  v7 = &v6[*(unsigned int *)(a1 + 20)];
  if ( v6 != v7 )
  {
    while ( a3 != *v6 )
    {
      if ( v7 == ++v6 )
        goto LABEL_10;
    }
    return 0;
  }
LABEL_10:
  if ( v4[28] )
  {
    v9 = (_QWORD *)*((_QWORD *)v4 + 1);
    v10 = &v9[*((unsigned int *)v4 + 5)];
    if ( v9 == v10 )
      return a3;
    while ( a3 != *v9 )
    {
      if ( v10 == ++v9 )
        return a3;
    }
  }
  else
  {
    a2 = (_BYTE *)a3;
    if ( !sub_C8CA60(v4, a3, v7, a4) )
      return a3;
  }
  if ( (unsigned __int8)(*(_BYTE *)a3 - 5) > 0x1Fu )
    return a3;
  v22 = 0;
  v11 = 0;
  v26 = 0x400000000LL;
  v12 = *(_BYTE *)(a3 - 16);
  v25 = v27;
  if ( (v12 & 2) == 0 )
    goto LABEL_25;
  while ( v11 < *(_DWORD *)(a3 - 24) )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 8LL * v11);
    if ( v13 )
    {
LABEL_19:
      if ( v13 == a3 )
      {
        v20 = (unsigned int)v26;
        if ( (unsigned __int64)(unsigned int)v26 + 1 > HIDWORD(v26) )
        {
          a2 = v27;
          sub_C8D5F0(&v25, v27, (unsigned int)v26 + 1LL, 8);
          v20 = (unsigned int)v26;
        }
        v22 = 1;
        *(_QWORD *)&v25[8 * v20] = 0;
        LODWORD(v26) = v26 + 1;
      }
      else
      {
        a2 = v4;
        v14 = sub_AE6BA0(a1, v4);
        if ( v14 )
        {
          v15 = (unsigned int)v26;
          if ( (unsigned __int64)(unsigned int)v26 + 1 > HIDWORD(v26) )
          {
            a2 = v27;
            v21 = v14;
            sub_C8D5F0(&v25, v27, (unsigned int)v26 + 1LL, 8);
            v15 = (unsigned int)v26;
            v14 = v21;
          }
          *(_QWORD *)&v25[8 * v15] = v14;
          LODWORD(v26) = v26 + 1;
        }
      }
      goto LABEL_24;
    }
    while ( 1 )
    {
      v16 = (unsigned int)v26;
      if ( (unsigned __int64)(unsigned int)v26 + 1 > HIDWORD(v26) )
      {
        a2 = v27;
        sub_C8D5F0(&v25, v27, (unsigned int)v26 + 1LL, 8);
        v16 = (unsigned int)v26;
      }
      *(_QWORD *)&v25[8 * v16] = 0;
      LODWORD(v26) = v26 + 1;
LABEL_24:
      v12 = *(_BYTE *)(a3 - 16);
      ++v11;
      if ( (v12 & 2) != 0 )
        break;
LABEL_25:
      if ( v11 >= ((*(_WORD *)(a3 - 16) >> 6) & 0xFu) )
        goto LABEL_32;
      v13 = *(_QWORD *)(a3 + -16 - 8LL * ((v12 >> 2) & 0xF) + 8LL * v11);
      if ( v13 )
        goto LABEL_19;
    }
  }
LABEL_32:
  result = 0;
  if ( (_DWORD)v26 && ((_DWORD)v26 != 1 || !v22) )
  {
    a2 = v25;
    v17 = *(_QWORD *)(a3 + 8);
    v18 = (_QWORD *)(v17 & 0xFFFFFFFFFFFFFFF8LL);
    v19 = (v17 >> 2) & 1;
    if ( (*(_BYTE *)(a3 + 1) & 0x7F) == 1 )
    {
      if ( (_BYTE)v19 )
        v18 = (_QWORD *)*v18;
      result = sub_B9C770(v18, v25, (unsigned int)v26, 1, 1);
    }
    else
    {
      if ( (_BYTE)v19 )
        v18 = (_QWORD *)*v18;
      result = sub_B9C770(v18, v25, (unsigned int)v26, 0, 1);
    }
    if ( v22 )
    {
      a2 = 0;
      v24 = result;
      sub_BA6610(result, 0, result);
      result = v24;
    }
  }
  if ( v25 != v27 )
  {
    v23 = result;
    _libc_free(v25, a2);
    return v23;
  }
  return result;
}
