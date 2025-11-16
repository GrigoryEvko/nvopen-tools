// Function: sub_2B10DF0
// Address: 0x2b10df0
//
__int64 __fastcall sub_2B10DF0(__int64 a1)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // r14
  __int64 i; // rax
  __int64 v7; // r8
  unsigned int v8; // edi
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // edx
  __int64 v13; // rdx
  _BYTE *v14; // r12
  __int64 v15; // rcx
  __int64 v16; // rsi

  v2 = **(_QWORD **)a1;
  v3 = **(_QWORD **)(a1 + 8);
  v4 = *(__int64 **)v3;
  v5 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  for ( i = *(_QWORD *)(v2 + 40); (__int64 *)v5 != v4; v2 = (__int64)v14 )
  {
    while ( 1 )
    {
      v14 = (_BYTE *)*v4;
      if ( *(_BYTE *)*v4 > 0x1Cu )
        break;
LABEL_11:
      if ( (__int64 *)v5 == ++v4 )
        goto LABEL_17;
    }
    v15 = *((_QWORD *)v14 + 5);
    if ( v15 == i )
    {
      if ( sub_B445A0(v2, *v4) )
      {
        i = *((_QWORD *)v14 + 5);
        v2 = (__int64)v14;
      }
      else
      {
        i = *(_QWORD *)(v2 + 40);
      }
      goto LABEL_11;
    }
    v16 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 3320LL);
    if ( i )
    {
      v7 = (unsigned int)(*(_DWORD *)(i + 44) + 1);
      v8 = *(_DWORD *)(v16 + 32);
      if ( *(_DWORD *)(i + 44) + 1 >= v8 )
        goto LABEL_16;
    }
    else
    {
      v8 = *(_DWORD *)(v16 + 32);
      v7 = 0;
      if ( !v8 )
        goto LABEL_16;
    }
    v9 = *(_QWORD *)(v16 + 24);
    v10 = *(_QWORD *)(v9 + 8 * v7);
    if ( v10 )
    {
      if ( v15 )
      {
        v11 = (unsigned int)(*(_DWORD *)(v15 + 44) + 1);
        v12 = *(_DWORD *)(v15 + 44) + 1;
      }
      else
      {
        v11 = 0;
        v12 = 0;
      }
      if ( v8 > v12 )
      {
        v13 = *(_QWORD *)(v9 + 8 * v11);
        if ( v13 )
        {
          if ( *(_DWORD *)(v10 + 72) < *(_DWORD *)(v13 + 72) )
          {
            i = *((_QWORD *)v14 + 5);
            v2 = *v4;
          }
        }
      }
      goto LABEL_11;
    }
LABEL_16:
    ++v4;
    i = *((_QWORD *)v14 + 5);
  }
LABEL_17:
  **(_QWORD **)(a1 + 24) = i;
  return v2;
}
