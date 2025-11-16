// Function: sub_1029DA0
// Address: 0x1029da0
//
__int64 __fastcall sub_1029DA0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // ecx
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 *v9; // rbx
  __int64 v10; // rdi
  __int64 result; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  int v14; // ecx
  __int64 v15; // rcx
  __int64 *v16; // rax
  int v17; // eax
  int v18; // r10d

  v6 = *(_DWORD *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 72LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v18 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v18 + v8);
      v9 = (__int64 *)(v7 + 72LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      ++v18;
    }
  }
  v9 = (__int64 *)(v7 + 72LL * v6);
LABEL_3:
  if ( *((_BYTE *)v9 + 36) )
  {
    result = *((unsigned int *)v9 + 7);
    v12 = v9[2];
    v13 = v12 + 8 * result;
    v14 = *((_DWORD *)v9 + 7);
    if ( v12 != v13 )
    {
      result = v9[2];
      while ( a3 != *(_QWORD *)result )
      {
        result += 8;
        if ( v13 == result )
          goto LABEL_11;
      }
      v15 = (unsigned int)(v14 - 1);
      *((_DWORD *)v9 + 7) = v15;
      *(_QWORD *)result = *(_QWORD *)(v12 + 8 * v15);
      result = *((unsigned int *)v9 + 8);
      ++v9[1];
LABEL_9:
      if ( *((_DWORD *)v9 + 7) != (_DWORD)result )
        return result;
      goto LABEL_15;
    }
LABEL_11:
    if ( v14 == *((_DWORD *)v9 + 8) )
    {
LABEL_12:
      *v9 = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
  }
  else
  {
    v16 = sub_C8CA60((__int64)(v9 + 1), a3);
    if ( !v16 )
    {
      result = *((unsigned int *)v9 + 8);
      goto LABEL_9;
    }
    *v16 = -2;
    v17 = *((_DWORD *)v9 + 8);
    ++v9[1];
    result = (unsigned int)(v17 + 1);
    *((_DWORD *)v9 + 8) = result;
    if ( *((_DWORD *)v9 + 7) == (_DWORD)result )
    {
LABEL_15:
      if ( !*((_BYTE *)v9 + 36) )
        result = _libc_free(v9[2], a3);
      goto LABEL_12;
    }
  }
  return result;
}
