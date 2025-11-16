// Function: sub_2157BA0
// Address: 0x2157ba0
//
__int64 __fastcall sub_2157BA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  int v4; // eax
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 i; // rax
  __int64 v13; // rdi
  __int64 *v14; // r9
  int v15; // edx
  int v16; // r10d
  int v17; // eax
  __int64 v18; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v19; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1;
  v4 = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)v4 == 3 )
  {
    v5 = *(_DWORD *)(a2 + 24);
    v18 = a1;
    if ( v5 )
    {
      v6 = *(_QWORD *)(a2 + 8);
      result = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = (__int64 *)(v6 + 8 * result);
      v9 = *v8;
      if ( v3 == *v8 )
        return result;
      v16 = 1;
      v14 = 0;
      while ( v9 != -8 )
      {
        if ( v14 || v9 != -16 )
          v8 = v14;
        result = (v5 - 1) & (v16 + (_DWORD)result);
        v9 = *(_QWORD *)(v6 + 8LL * (unsigned int)result);
        if ( v3 == v9 )
          return result;
        ++v16;
        v14 = v8;
        v8 = (__int64 *)(v6 + 8LL * (unsigned int)result);
      }
      v17 = *(_DWORD *)(a2 + 16);
      if ( !v14 )
        v14 = v8;
      ++*(_QWORD *)a2;
      v15 = v17 + 1;
      if ( 4 * (v17 + 1) < 3 * v5 )
      {
        result = v5 - *(_DWORD *)(a2 + 20) - v15;
        if ( (unsigned int)result > v5 >> 3 )
          goto LABEL_15;
        goto LABEL_14;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    v5 *= 2;
LABEL_14:
    sub_21579F0(a2, v5);
    sub_21552D0(a2, &v18, &v19);
    result = *(unsigned int *)(a2 + 16);
    v14 = v19;
    v3 = v18;
    v15 = result + 1;
LABEL_15:
    *(_DWORD *)(a2 + 16) = v15;
    if ( *v14 != -8 )
      --*(_DWORD *)(a2 + 20);
    *v14 = v3;
    return result;
  }
  result = (unsigned int)(v4 - 17);
  v18 = 0;
  if ( (unsigned __int8)result > 6u )
  {
    result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      v10 = 0;
      v11 = 24LL * (unsigned int)result;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
        goto LABEL_11;
LABEL_8:
      for ( i = *(_QWORD *)(v3 - 8); ; i = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) )
      {
        v13 = *(_QWORD *)(i + v10);
        v10 += 24;
        result = sub_2157BA0(v13, a2);
        if ( v11 == v10 )
          break;
        if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
          goto LABEL_8;
LABEL_11:
        ;
      }
    }
  }
  return result;
}
