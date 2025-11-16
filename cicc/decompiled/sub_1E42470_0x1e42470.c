// Function: sub_1E42470
// Address: 0x1e42470
//
__int64 __fastcall sub_1E42470(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 i; // r8
  __int64 *v6; // rdi
  __int64 *v7; // rdx
  __int64 v8; // rbx
  __int64 *v9; // rcx
  __int64 *v10; // rax
  __int64 *v11; // rsi
  __int64 *v12; // rax
  bool v13; // zf
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rcx

  *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * (a2 >> 6)) &= ~(1LL << a2);
  result = *(_QWORD *)(a1 + 88);
  v3 = result + 72LL * (int)a2;
  for ( i = *(unsigned int *)(v3 + 28); (_DWORD)i != *(_DWORD *)(v3 + 32); i = *(unsigned int *)(v3 + 28) )
  {
    v6 = *(__int64 **)(v3 + 16);
    v7 = *(__int64 **)(v3 + 8);
    v8 = *v6;
    if ( v6 == v7 )
    {
      v16 = (unsigned int)i;
      v9 = &v6[v16];
      if ( v6 == &v6[v16] )
        goto LABEL_16;
    }
    else
    {
      v9 = &v6[*(unsigned int *)(v3 + 24)];
      if ( v6 == v9 )
        goto LABEL_9;
    }
    v10 = *(__int64 **)(v3 + 16);
    while ( 1 )
    {
      v8 = *v10;
      v11 = v10;
      if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v9 == ++v10 )
      {
        v8 = v11[1];
        break;
      }
    }
    if ( v6 == v7 )
    {
      v16 = i;
LABEL_16:
      v17 = &v7[v16];
      if ( v7 == &v7[v16] )
      {
LABEL_28:
        v7 = &v6[v16];
        v18 = &v6[*(unsigned int *)(v3 + 28)];
      }
      else
      {
        while ( v8 != *v7 )
        {
          if ( v17 == ++v7 )
            goto LABEL_28;
        }
        v18 = &v6[*(unsigned int *)(v3 + 28)];
      }
LABEL_21:
      if ( v18 != v7 )
      {
        *v7 = -2;
        ++*(_DWORD *)(v3 + 32);
      }
      goto LABEL_11;
    }
LABEL_9:
    v12 = sub_16CC9F0(v3, v8);
    v13 = *v12 == v8;
    v7 = v12;
    v14 = *(_QWORD *)(v3 + 16);
    if ( v13 )
    {
      if ( v14 == *(_QWORD *)(v3 + 8) )
        v19 = *(unsigned int *)(v3 + 28);
      else
        v19 = *(unsigned int *)(v3 + 24);
      v18 = (__int64 *)(v14 + 8 * v19);
      goto LABEL_21;
    }
    if ( v14 == *(_QWORD *)(v3 + 8) )
    {
      v7 = (__int64 *)(v14 + 8LL * *(unsigned int *)(v3 + 28));
      v18 = v7;
      goto LABEL_21;
    }
LABEL_11:
    v15 = *(_DWORD *)(v8 + 192);
    result = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * (v15 >> 6)) & (1LL << v15);
    if ( result )
      result = sub_1E42470(a1, v15);
  }
  return result;
}
