// Function: sub_1A826A0
// Address: 0x1a826a0
//
_QWORD *__fastcall sub_1A826A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v6; // edx
  _QWORD *result; // rax
  unsigned int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdx
  _QWORD *v14; // rbx
  _QWORD *v15; // rsi
  _QWORD *v16; // r12
  unsigned __int64 v17; // r13
  __int64 v18; // rcx

  v6 = *(unsigned __int8 *)(a2 + 16);
  result = a1;
  if ( (unsigned __int8)v6 <= 0x17u )
    v8 = *(unsigned __int16 *)(a2 + 18);
  else
    v8 = v6 - 24;
  if ( v8 == 53 )
  {
    v13 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v14 = (_QWORD *)(a2 - v13);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v14 = *(_QWORD **)(a2 - 8);
    v15 = a1 + 2;
    *a1 = a1 + 2;
    v16 = &v14[(unsigned __int64)v13 / 8];
    v17 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
    a1[1] = 0x200000000LL;
    LODWORD(v18) = 0;
    if ( (unsigned __int64)v13 > 0x30 )
    {
      sub_16CD150((__int64)a1, v15, 0xAAAAAAAAAAAAAAABLL * (v13 >> 3), 8, a5, a6);
      result = a1;
      v18 = *((unsigned int *)a1 + 2);
      v15 = (_QWORD *)(*a1 + 8 * v18);
    }
    if ( v16 != v14 )
    {
      do
      {
        if ( v15 )
          *v15 = *v14;
        v14 += 3;
        ++v15;
      }
      while ( v14 != v16 );
      LODWORD(v18) = *((_DWORD *)result + 2);
    }
    *((_DWORD *)result + 2) = v18 + v17;
  }
  else if ( v8 <= 0x35 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v11 = *(__int64 **)(a2 - 8);
    else
      v11 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v12 = *v11;
    *a1 = a1 + 2;
    a1[2] = v12;
    a1[1] = 0x200000001LL;
  }
  else
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v9 = *(_QWORD *)(a2 - 8);
    else
      v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v10 = *(_QWORD *)(v9 + 48);
    a1[2] = *(_QWORD *)(v9 + 24);
    *a1 = a1 + 2;
    a1[3] = v10;
    a1[1] = 0x200000002LL;
  }
  return result;
}
