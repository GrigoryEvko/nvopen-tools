// Function: sub_1374B30
// Address: 0x1374b30
//
__int64 __fastcall sub_1374B30(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // esi
  __int64 *v5; // rdx
  __int64 v6; // r9
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rsi
  int v13; // edx
  int v14; // r10d

  result = *(_QWORD *)(a1 + 96);
  v2 = *(__int64 **)(result - 40);
  if ( *(__int64 **)(*(_QWORD *)(result - 48) + 56LL) != v2 )
  {
    while ( 1 )
    {
      *(_QWORD *)(result - 40) = v2 + 1;
      if ( v2 + 1 == *(__int64 **)(result - 24) )
      {
        v10 = (__int64 *)(*(_QWORD *)(result - 16) + 8LL);
        *(_QWORD *)(result - 16) = v10;
        v11 = *v10;
        v12 = *v10 + 512;
        *(_QWORD *)(result - 32) = v11;
        *(_QWORD *)(result - 24) = v12;
        *(_QWORD *)(result - 40) = v11;
      }
      v8 = *(unsigned int *)(a1 + 32);
      v9 = *v2;
      if ( !(_DWORD)v8 )
        goto LABEL_11;
      v3 = *(_QWORD *)(a1 + 16);
      v4 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( v9 != *v5 )
      {
        v13 = 1;
        while ( v6 != -8 )
        {
          v14 = v13 + 1;
          v4 = (v8 - 1) & (v13 + v4);
          v5 = (__int64 *)(v3 + 16LL * v4);
          v6 = *v5;
          if ( v9 == *v5 )
            goto LABEL_4;
          v13 = v14;
        }
        goto LABEL_11;
      }
LABEL_4:
      if ( v5 == (__int64 *)(v3 + 16 * v8) )
      {
LABEL_11:
        sub_13747A0((int *)a1, v9);
        result = *(_QWORD *)(a1 + 96);
        v2 = *(__int64 **)(result - 40);
        if ( *(__int64 **)(*(_QWORD *)(result - 48) + 56LL) == v2 )
          return result;
      }
      else
      {
        result = *(_QWORD *)(a1 + 96);
        v7 = *((_DWORD *)v5 + 2);
        if ( *(_DWORD *)(result - 8) > v7 )
        {
          *(_DWORD *)(result - 8) = v7;
          result = *(_QWORD *)(a1 + 96);
        }
        v2 = *(__int64 **)(result - 40);
        if ( *(__int64 **)(*(_QWORD *)(result - 48) + 56LL) == v2 )
          return result;
      }
    }
  }
  return result;
}
