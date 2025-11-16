// Function: sub_141AD70
// Address: 0x141ad70
//
__int64 __fastcall sub_141AD70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 v10; // rdi
  int v11; // esi
  int v12; // r10d
  __int64 *v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rdi
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  int v20; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 88LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 352;
  }
  for ( i = result + v8; i != result; result += 88 )
  {
    if ( result )
      *(_QWORD *)result = -8;
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -16 && result != -8 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 3;
        }
        else
        {
          v20 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v20 )
          {
            MEMORY[0] = *(_QWORD *)v5;
            BUG();
          }
          v11 = v20 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v15 = (__int64 *)(v10 + 88LL * v14);
        v16 = *v15;
        if ( result != *v15 )
        {
          while ( v16 != -8 )
          {
            if ( v16 == -16 && !v13 )
              v13 = v15;
            v14 = v11 & (v12 + v14);
            v15 = (__int64 *)(v10 + 88LL * v14);
            v16 = *v15;
            if ( result == *v15 )
              goto LABEL_13;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_13:
        *v15 = result;
        v17 = (__int64)(v15 + 1);
        v18 = v15 + 3;
        v19 = v15 + 11;
        *(v19 - 10) = 0;
        *(v19 - 9) = 1;
        do
        {
          if ( v18 )
            *v18 = -8;
          v18 += 2;
        }
        while ( v18 != v19 );
        sub_1415470(v17, v5 + 8);
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
        if ( (*(_BYTE *)(v5 + 16) & 1) == 0 )
          result = j___libc_free_0(*(_QWORD *)(v5 + 24));
      }
      v5 += 88;
    }
    while ( a3 != v5 );
  }
  return result;
}
