// Function: sub_11AF9F0
// Address: 0x11af9f0
//
__int64 __fastcall sub_11AF9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  unsigned int v8; // r8d
  __int64 v10; // rcx
  __int64 v11; // r14
  _BYTE *v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // r9
  __int64 v15; // rax
  int v16; // r13d
  unsigned int v17; // ecx
  __int64 v18; // rax
  bool v19; // cl
  bool v20; // si
  _DWORD *v21; // rax
  __int64 v22; // rax
  int v23; // r13d
  const void *v24; // r14
  __int64 v25; // rax
  const void *v26; // r14
  int v27; // r13d
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // r13
  int v32; // [rsp+0h] [rbp-40h]

  v7 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  if ( *(_BYTE *)a1 == 13 )
  {
    if ( *(_DWORD *)(a4 + 12) < v7 )
    {
      *(_DWORD *)(a4 + 8) = 0;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v7, 4u, a5, a6);
      memset(*(void **)a4, 255, 4LL * v7);
      *(_DWORD *)(a4 + 8) = v7;
    }
    else
    {
      v29 = *(unsigned int *)(a4 + 8);
      v30 = v29;
      if ( v7 <= v29 )
        v30 = v7;
      if ( v30 )
      {
        memset(*(void **)a4, 255, 4 * v30);
        v29 = *(unsigned int *)(a4 + 8);
      }
      if ( v7 > v29 )
      {
        v31 = v7 - v29;
        if ( v31 )
        {
          if ( 4 * v31 )
            memset((void *)(*(_QWORD *)a4 + 4 * v29), 255, 4 * v31);
        }
      }
      *(_DWORD *)(a4 + 8) = v7;
    }
    return 1;
  }
  if ( a1 == a2 )
  {
    if ( v7 )
    {
      v22 = *(unsigned int *)(a4 + 8);
      v23 = 0;
      v24 = (const void *)(a4 + 16);
      do
      {
        if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          sub_C8D5F0(a4, v24, v22 + 1, 4u, a5, a6);
          v22 = *(unsigned int *)(a4 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a4 + 4 * v22) = v23++;
        v22 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
        *(_DWORD *)(a4 + 8) = v22;
      }
      while ( v7 != v23 );
    }
    return 1;
  }
  if ( a1 == a3 )
  {
    if ( v7 )
    {
      v25 = *(unsigned int *)(a4 + 8);
      v26 = (const void *)(a4 + 16);
      v27 = 2 * v7;
      v28 = v25 + 1;
      if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        goto LABEL_31;
      while ( 1 )
      {
        *(_DWORD *)(*(_QWORD *)a4 + 4 * v25) = v7++;
        v25 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
        *(_DWORD *)(a4 + 8) = v25;
        if ( v27 == v7 )
          break;
        v28 = v25 + 1;
        if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
LABEL_31:
          sub_C8D5F0(a4, v26, v28, 4u, a5, a6);
          v25 = *(unsigned int *)(a4 + 8);
        }
      }
    }
    return 1;
  }
  if ( *(_BYTE *)a1 != 91 )
    return 0;
  v10 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v10 != 17 )
    return 0;
  v11 = *(_QWORD *)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = **(_QWORD **)(v10 + 24);
  v12 = *(_BYTE **)(a1 - 64);
  if ( *v12 == 13 )
  {
    v8 = sub_11AF9F0(*(_QWORD *)(a1 - 96), a2, a3, a4);
    if ( (_BYTE)v8 )
    {
      *(_DWORD *)(*(_QWORD *)a4 + 4LL * (unsigned int)v11) = -1;
      return v8;
    }
    return 0;
  }
  if ( *v12 != 90 )
    return 0;
  v13 = *((_QWORD *)v12 - 4);
  if ( *(_BYTE *)v13 != 17 )
    return 0;
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = *((_QWORD *)v12 - 8);
  if ( (!v15 || a2 != v15) && a3 != v15 )
    return 0;
  v32 = (int)v14;
  v16 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL);
  v8 = sub_11AF9F0(*(_QWORD *)(a1 - 96), a2, a3, a4);
  if ( !(_BYTE)v8 )
    return 0;
  v18 = *((_QWORD *)v12 - 8);
  v19 = a2 == v18;
  v20 = v18 != 0;
  v21 = (_DWORD *)(*(_QWORD *)a4 + 4LL * ((unsigned int)v11 % v7));
  LOBYTE(v17) = v20 && v19;
  if ( (_BYTE)v17 )
  {
    *v21 = v32;
    return v17;
  }
  else
  {
    *v21 = v16 + v32;
  }
  return v8;
}
