// Function: sub_35431F0
// Address: 0x35431f0
//
__int64 __fastcall sub_35431F0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  int v5; // edi
  __int64 *v6; // rsi
  char v7; // r8
  __int64 v8; // rbx
  __int64 *v9; // rdx
  __int64 *v10; // rax
  __int64 *v11; // rcx
  __int64 *v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // rax

  result = ~(1LL << a2);
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL * (a2 >> 6)) &= result;
  v4 = *(_QWORD *)(a1 + 128) + ((__int64)(int)a2 << 6);
  v5 = *(_DWORD *)(v4 + 20);
  if ( *(_DWORD *)(v4 + 24) != v5 )
  {
    while ( 1 )
    {
      v6 = *(__int64 **)(v4 + 8);
      v7 = *(_BYTE *)(v4 + 28);
      v8 = *v6;
      if ( !v7 )
        break;
      v9 = &v6[v5];
      if ( v6 != v9 )
        goto LABEL_4;
LABEL_14:
      v15 = *(_DWORD *)(v8 + 200);
      result = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL * (v15 >> 6)) & (1LL << v15);
      if ( result )
        result = sub_35431F0(a1, v15);
      v5 = *(_DWORD *)(v4 + 20);
      if ( v5 == *(_DWORD *)(v4 + 24) )
        return result;
    }
    v9 = &v6[*(unsigned int *)(v4 + 16)];
    if ( v6 != v9 )
    {
LABEL_4:
      v10 = *(__int64 **)(v4 + 8);
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
      if ( v7 )
      {
        v12 = &v6[v5];
        if ( v6 != v12 )
        {
          v13 = *(_QWORD **)(v4 + 8);
          while ( *v13 != v8 )
          {
            if ( ++v13 == v12 )
              goto LABEL_14;
          }
          v14 = (unsigned int)(v5 - 1);
          *(_DWORD *)(v4 + 20) = v14;
          *v13 = v6[v14];
          ++*(_QWORD *)v4;
        }
        goto LABEL_14;
      }
    }
    v16 = sub_C8CA60(v4, v8);
    if ( v16 )
    {
      *v16 = -2;
      ++*(_DWORD *)(v4 + 24);
      ++*(_QWORD *)v4;
    }
    goto LABEL_14;
  }
  return result;
}
