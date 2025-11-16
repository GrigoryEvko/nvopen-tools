// Function: sub_B98540
// Address: 0xb98540
//
__int64 __fastcall sub_B98540(_WORD *a1, __int64 a2)
{
  unsigned __int8 **v2; // r12
  __int64 *v3; // rdx
  __int64 *v4; // rax
  unsigned __int8 **v5; // rbx
  __int64 v6; // r13
  unsigned __int8 *v7; // rsi
  __int64 v8; // rsi
  unsigned int v9; // r15d
  __int64 *v10; // r12
  __int64 result; // rax
  __int64 *v12; // rbx
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 *v16; // [rsp+0h] [rbp-40h] BYREF
  __int64 v17; // [rsp+8h] [rbp-38h]
  _BYTE v18[48]; // [rsp+10h] [rbp-30h] BYREF

  v2 = (unsigned __int8 **)v18;
  v16 = (__int64 *)v18;
  v17 = 0;
  if ( a2 )
  {
    sub_B97700((__int64)&v16, a2);
    v2 = (unsigned __int8 **)v16;
    v3 = &v16[a2];
    v4 = &v16[(unsigned int)v17];
    if ( v4 != v3 )
    {
      do
      {
        if ( v4 )
          *v4 = 0;
        ++v4;
      }
      while ( v3 != v4 );
      v2 = (unsigned __int8 **)v16;
    }
    LODWORD(v17) = a2;
  }
  if ( (*(_BYTE *)a1 & 2) != 0 )
  {
    v5 = (unsigned __int8 **)*((_QWORD *)a1 - 2);
    v6 = *((unsigned int *)a1 - 2);
  }
  else
  {
    v6 = (*a1 >> 6) & 0xF;
    v5 = (unsigned __int8 **)&a1[-4 * ((*(_BYTE *)a1 >> 2) & 0xF)];
  }
  for ( ; v6; --v6 )
  {
    v7 = *v5;
    *v2 = *v5;
    if ( v7 )
      sub_B976B0((__int64)v5, v7, (__int64)v2);
    *v5 = 0;
    ++v2;
    ++v5;
  }
  v8 = 0;
  sub_B91520(a1, 0);
  v9 = v17;
  *((_QWORD *)a1 - 2) = a1;
  *((_DWORD *)a1 - 2) = 0;
  *((_DWORD *)a1 - 1) = 0;
  if ( !v9 )
  {
LABEL_15:
    v10 = v16;
    goto LABEL_16;
  }
  if ( v16 != (__int64 *)v18 )
  {
    *((_QWORD *)a1 - 2) = v16;
    result = HIDWORD(v17);
    *((_DWORD *)a1 - 2) = v9;
    *(_BYTE *)a1 |= 2u;
    *((_DWORD *)a1 - 1) = result;
    return result;
  }
  v8 = v9;
  sub_B97700((__int64)(a1 - 8), v9);
  v12 = v16;
  v13 = (__int64 *)*((_QWORD *)a1 - 2);
  v10 = &v16[(unsigned int)v17];
  if ( v16 != v10 )
  {
    do
    {
      if ( v13 )
      {
        v8 = *v12;
        *v13 = *v12;
        if ( v8 )
          sub_B976B0((__int64)v12, (unsigned __int8 *)v8, (__int64)v13);
        *v12 = 0;
      }
      ++v12;
      ++v13;
    }
    while ( v10 != v12 );
    v10 = v16;
    v14 = (unsigned int)v17;
    *((_DWORD *)a1 - 2) = v9;
    v15 = (__int64)&v10[v14];
    if ( v10 == (__int64 *)v15 )
      goto LABEL_16;
    do
    {
      v8 = *(_QWORD *)(v15 - 8);
      v15 -= 8;
      if ( v8 )
        sub_B91220(v15, v8);
    }
    while ( v10 != (__int64 *)v15 );
    goto LABEL_15;
  }
  *((_DWORD *)a1 - 2) = v9;
LABEL_16:
  result = (__int64)v18;
  *(_BYTE *)a1 |= 2u;
  if ( v10 != (__int64 *)v18 )
    return _libc_free(v10, v8);
  return result;
}
