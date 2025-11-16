// Function: sub_2ECB100
// Address: 0x2ecb100
//
__int64 __fastcall sub_2ECB100(__int64 a1)
{
  _QWORD *v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // r15
  unsigned __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r9
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rax
  int v16; // edx
  int v17; // ecx
  unsigned __int64 v18; // rcx
  __int64 result; // rax
  int v20; // edx
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  _BYTE *v23; // rax
  _BYTE *v24; // rax

  v2 = *(_QWORD **)(a1 + 3552);
  if ( v2 )
  {
    v3 = v2[1];
    v4 = v2[2];
    v5 = (unsigned __int64 *)v2[22];
    v6 = (unsigned __int64 *)v2[23];
  }
  else
  {
    v23 = (_BYTE *)sub_22077B0(0xE0u);
    v2 = v23;
    if ( v23 )
    {
      *v23 = 1;
      v24 = v23 + 48;
      v6 = 0;
      v5 = 0;
      *((_DWORD *)v24 - 11) = 8;
      v4 = 0;
      *((_QWORD *)v24 - 5) = 0;
      *((_QWORD *)v24 - 4) = 0;
      *((_QWORD *)v24 - 3) = 0;
      v2[22] = 0;
      v2[23] = 0;
      v2[24] = 0;
      v2[25] = 0;
      v2[26] = 0;
      v2[27] = 0;
      v2[4] = v24;
      v2[5] = 0x1000000000LL;
      v3 = 0;
    }
    else
    {
      v3 = MEMORY[8];
      v4 = MEMORY[0x10];
      v5 = (unsigned __int64 *)MEMORY[0xB0];
      v6 = (unsigned __int64 *)MEMORY[0xB8];
    }
    *(_QWORD *)(a1 + 3552) = v2;
  }
  if ( v3 != v4 )
    v2[2] = v3;
  *((_DWORD *)v2 + 10) = 0;
  if ( v5 != v6 )
  {
    v7 = v5;
    do
    {
      if ( (unsigned __int64 *)*v7 != v7 + 2 )
        _libc_free(*v7);
      v7 += 6;
    }
    while ( v7 != v6 );
    v2[23] = v5;
  }
  v8 = v2[25];
  if ( v8 != v2[26] )
    v2[26] = v8;
  v9 = *(_QWORD *)(a1 + 3552);
  v10 = *(_QWORD *)(a1 + 48);
  *(_DWORD *)(a1 + 3624) = 0;
  *(_DWORD *)(a1 + 3568) = 0;
  v11 = *(_QWORD *)(v9 + 8);
  v12 = (unsigned int)((*(_QWORD *)(a1 + 56) - v10) >> 8);
  v13 = (*(_QWORD *)(v9 + 16) - v11) >> 3;
  if ( v12 > v13 )
  {
    sub_2ECAF50((_QWORD *)(v9 + 8), v12 - v13);
  }
  else if ( v12 < v13 )
  {
    v22 = v11 + 8 * v12;
    if ( *(_QWORD *)(v9 + 16) != v22 )
      *(_QWORD *)(v9 + 16) = v22;
  }
  sub_2F95AC0();
  v15 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 3552) + 208LL) - *(_QWORD *)(*(_QWORD *)(a1 + 3552) + 200LL)) >> 2;
  LOBYTE(v16) = v15;
  v17 = *(_DWORD *)(a1 + 3624) & 0x3F;
  if ( v17 )
    *(_QWORD *)(*(_QWORD *)(a1 + 3560) + 8LL * *(unsigned int *)(a1 + 3568) - 8) &= ~(-1LL << v17);
  *(_DWORD *)(a1 + 3624) = v15;
  v18 = *(unsigned int *)(a1 + 3568);
  result = (unsigned int)(v15 + 63) >> 6;
  if ( (unsigned int)result != v18 )
  {
    if ( (unsigned int)result >= v18 )
    {
      v21 = (unsigned int)result - v18;
      if ( (unsigned int)result > (unsigned __int64)*(unsigned int *)(a1 + 3572) )
      {
        sub_C8D5F0(a1 + 3560, (const void *)(a1 + 3576), (unsigned int)result, 8u, (unsigned int)result, v14);
        v18 = *(unsigned int *)(a1 + 3568);
      }
      result = *(_QWORD *)(a1 + 3560);
      if ( 8 * v21 )
      {
        result = (__int64)memset((void *)(result + 8 * v18), 0, 8 * v21);
        LODWORD(v18) = *(_DWORD *)(a1 + 3568);
      }
      v16 = *(_DWORD *)(a1 + 3624);
      *(_DWORD *)(a1 + 3568) = v21 + v18;
    }
    else
    {
      *(_DWORD *)(a1 + 3568) = result;
    }
  }
  v20 = v16 & 0x3F;
  if ( v20 )
  {
    result = ~(-1LL << v20);
    *(_QWORD *)(*(_QWORD *)(a1 + 3560) + 8LL * *(unsigned int *)(a1 + 3568) - 8) &= result;
  }
  return result;
}
