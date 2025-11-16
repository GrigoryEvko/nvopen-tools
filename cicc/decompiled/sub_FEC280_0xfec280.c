// Function: sub_FEC280
// Address: 0xfec280
//
__int64 __fastcall sub_FEC280(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  unsigned int v9; // edx
  __int64 *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rsi
  int v13; // edx
  int v14; // r10d

  result = *(_QWORD *)(a1 + 96);
  v2 = *(__int64 **)(result - 40);
  if ( v2 != *(__int64 **)(*(_QWORD *)(result - 48) + 56LL) )
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
        v3 = *(unsigned int *)(a1 + 32);
        v4 = *v2;
        v5 = *(_QWORD *)(a1 + 16);
        if ( !(_DWORD)v3 )
          goto LABEL_11;
      }
      else
      {
        v3 = *(unsigned int *)(a1 + 32);
        v4 = *v2;
        v5 = *(_QWORD *)(a1 + 16);
        if ( !(_DWORD)v3 )
          goto LABEL_11;
      }
      v6 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( v4 != *v7 )
      {
        v13 = 1;
        while ( v8 != -4096 )
        {
          v14 = v13 + 1;
          v6 = (v3 - 1) & (v13 + v6);
          v7 = (__int64 *)(v5 + 16LL * v6);
          v8 = *v7;
          if ( v4 == *v7 )
            goto LABEL_5;
          v13 = v14;
        }
        goto LABEL_11;
      }
LABEL_5:
      if ( v7 == (__int64 *)(v5 + 16 * v3) )
      {
LABEL_11:
        sub_FEBEE0((int *)a1, v4);
        result = *(_QWORD *)(a1 + 96);
        v2 = *(__int64 **)(result - 40);
        if ( *(__int64 **)(*(_QWORD *)(result - 48) + 56LL) == v2 )
          return result;
      }
      else
      {
        result = *(_QWORD *)(a1 + 96);
        v9 = *((_DWORD *)v7 + 2);
        if ( *(_DWORD *)(result - 8) > v9 )
        {
          *(_DWORD *)(result - 8) = v9;
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
