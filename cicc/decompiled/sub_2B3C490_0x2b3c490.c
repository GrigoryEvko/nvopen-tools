// Function: sub_2B3C490
// Address: 0x2b3c490
//
__int64 __fastcall sub_2B3C490(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdx
  unsigned int v11; // r15d
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 j; // rcx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 k; // rdx
  unsigned __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rsi
  int v24; // edi
  __int64 *i; // rax
  __int64 *v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 v28; // [rsp+8h] [rbp-38h]

  v10 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)a1 == 91 )
  {
    v11 = *(_DWORD *)(v10 + 32);
  }
  else
  {
    v11 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v20 = *(unsigned __int8 *)(v10 + 8);
        if ( (_BYTE)v20 != 15 )
          break;
        v21 = *(__int64 **)(v10 + 16);
        v22 = *(unsigned int *)(v10 + 12);
        v23 = &v21[v22];
        v24 = v22;
        v10 = *v21;
        if ( v21 != v23 )
        {
          for ( i = v21 + 1; ; ++i )
          {
            v26 = i;
            if ( v23 == i )
              break;
            if ( v10 != *v26 )
            {
              LODWORD(a5) = 0;
              return (unsigned int)a5;
            }
          }
        }
        v11 *= v24;
      }
      if ( (_BYTE)v20 != 16 )
        break;
      v11 *= *(_DWORD *)(v10 + 32);
      v10 = *(_QWORD *)(v10 + 24);
    }
    if ( (_BYTE)v20 == 17 )
    {
      v11 *= *(_DWORD *)(v10 + 32);
    }
    else if ( (unsigned __int8)v20 > 3u && (_BYTE)v20 != 5 )
    {
      a5 = 0;
      if ( (unsigned __int8)v20 > 0x14u )
        return (unsigned int)a5;
      v27 = 1463376;
      if ( !_bittest64(&v27, v20) )
        return (unsigned int)a5;
    }
  }
  v12 = *(unsigned int *)(a2 + 8);
  v13 = v11;
  if ( v11 != v12 )
  {
    if ( v11 >= v12 )
    {
      if ( v11 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v11, 8u, a5, a6);
        v12 = *(unsigned int *)(a2 + 8);
        v13 = v11;
      }
      v14 = (_QWORD *)(*(_QWORD *)a2 + 8 * v12);
      for ( j = *(_QWORD *)a2 + 8 * v13; (_QWORD *)j != v14; ++v14 )
      {
        if ( v14 )
          *v14 = 0;
      }
    }
    *(_DWORD *)(a2 + 8) = v11;
  }
  v16 = *(unsigned int *)(a3 + 8);
  if ( v13 != v16 )
  {
    if ( v13 >= v16 )
    {
      if ( v13 > *(unsigned int *)(a3 + 12) )
      {
        v28 = v13;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v13, 8u, a5, a6);
        v16 = *(unsigned int *)(a3 + 8);
        v13 = v28;
      }
      v17 = (_QWORD *)(*(_QWORD *)a3 + 8 * v16);
      for ( k = *(_QWORD *)a3 + 8 * v13; (_QWORD *)k != v17; ++v17 )
      {
        if ( v17 )
          *v17 = 0;
      }
    }
    *(_DWORD *)(a3 + 8) = v11;
  }
  sub_2B18FA0((unsigned __int8 *)a1, (_QWORD *)a2, (_QWORD *)a3, 0, a4);
  sub_2B3C320(a2);
  sub_2B3C320(a3);
  LOBYTE(a5) = *(_DWORD *)(a2 + 8) > 1u;
  return (unsigned int)a5;
}
