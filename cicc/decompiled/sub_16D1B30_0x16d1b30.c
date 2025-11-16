// Function: sub_16D1B30
// Address: 0x16d1b30
//
__int64 __fastcall sub_16D1B30(__int64 *a1, unsigned __int8 *a2, size_t a3)
{
  unsigned int v4; // edi
  unsigned __int8 *v6; // rsi
  unsigned int v7; // r15d
  unsigned __int8 *v9; // rax
  unsigned int v10; // ebx
  int v11; // ecx
  unsigned int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  unsigned int v16; // r9d
  _QWORD *v17; // rax
  int v19; // r9d
  __int64 v20; // r10
  unsigned int v21; // ecx
  int v22; // eax
  unsigned int v23; // [rsp+8h] [rbp-48h]
  int v24; // [rsp+Ch] [rbp-44h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v4 = *((_DWORD *)a1 + 2);
  if ( v4 )
  {
    v6 = &a2[a3];
    v7 = v4 - 1;
    if ( v6 == a2 )
    {
      v12 = 0;
      v14 = 0;
      v13 = 0;
      v10 = 0;
    }
    else
    {
      v9 = a2;
      v10 = 0;
      do
      {
        v11 = *v9++;
        v10 += v11 + 32 * v10;
      }
      while ( v6 != v9 );
      v12 = v7 & v10;
      v13 = v7 & v10;
      v14 = 8 * v13;
    }
    v15 = *a1;
    v16 = -1;
    v17 = *(_QWORD **)(*a1 + v14);
    if ( v17 )
    {
      v19 = 1;
      v20 = 8LL * v4 + 8;
      while ( 1 )
      {
        if ( v17 != (_QWORD *)-8LL && *(_DWORD *)(v15 + 4 * v13 + v20) == v10 && *v17 == a3 )
        {
          v24 = v19;
          v25 = v20;
          v26 = v15;
          if ( !a3 )
            break;
          v23 = v12;
          v22 = memcmp(a2, (char *)v17 + *((unsigned int *)a1 + 5), a3);
          v12 = v23;
          v15 = v26;
          v20 = v25;
          v19 = v24;
          if ( !v22 )
            break;
        }
        v21 = v19 + v12;
        ++v19;
        v12 = v7 & v21;
        v13 = v12;
        v17 = *(_QWORD **)(v15 + 8LL * v12);
        if ( !v17 )
          return (unsigned int)-1;
      }
      return v12;
    }
  }
  else
  {
    return (unsigned int)-1;
  }
  return v16;
}
