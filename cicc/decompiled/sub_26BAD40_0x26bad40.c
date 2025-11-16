// Function: sub_26BAD40
// Address: 0x26bad40
//
__int64 __fastcall sub_26BAD40(__int64 a1, _QWORD *a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // r9d
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // r10
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 result; // rax
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rdx
  _QWORD *v17; // rcx
  unsigned __int64 v18; // rdi
  int v19; // eax
  int v20; // r11d

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = (v2 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == *a2 )
    {
LABEL_3:
      *v6 = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v19 = 1;
      while ( v7 != -4096 )
      {
        v20 = v19 + 1;
        v5 = v4 & (v19 + v5);
        v6 = (__int64 *)(v3 + 16LL * v5);
        v7 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_3;
        v19 = v20;
      }
    }
  }
  v8 = *(_QWORD *)(a1 + 32);
  v9 = a2 + 2;
  v10 = *(unsigned int *)(a1 + 40);
  v11 = (v8 + 16 * v10 - (__int64)(a2 + 2)) >> 4;
  if ( v8 + 16 * v10 - (__int64)(a2 + 2) > 0 )
  {
    do
    {
      v12 = *v9;
      v9 += 2;
      *(v9 - 4) = v12;
      *(v9 - 3) = *(v9 - 1);
      --v11;
    }
    while ( v11 );
    v8 = *(_QWORD *)(a1 + 32);
    LODWORD(v10) = *(_DWORD *)(a1 + 40);
  }
  v13 = (unsigned int)(v10 - 1);
  *(_DWORD *)(a1 + 40) = v13;
  result = v8 + 16 * v13;
  if ( a2 != (_QWORD *)result )
  {
    result = *(unsigned int *)(a1 + 16);
    v15 = ((__int64)a2 - v8) >> 4;
    if ( (_DWORD)result )
    {
      v16 = *(_QWORD **)(a1 + 8);
      v17 = &v16[2 * *(unsigned int *)(a1 + 24)];
      if ( v16 != v17 )
      {
        while ( 1 )
        {
          result = (__int64)v16;
          if ( *v16 != -4096 && *v16 != -8192 )
            break;
          v16 += 2;
          if ( v17 == v16 )
            return result;
        }
        if ( v16 != v17 )
        {
          do
          {
            v18 = *(unsigned int *)(result + 8);
            if ( v15 < v18 )
              *(_DWORD *)(result + 8) = v18 - 1;
            result += 16;
            if ( (_QWORD *)result == v17 )
              break;
            while ( *(_QWORD *)result == -8192 || *(_QWORD *)result == -4096 )
            {
              result += 16;
              if ( v17 == (_QWORD *)result )
                return result;
            }
          }
          while ( (_QWORD *)result != v17 );
        }
      }
    }
  }
  return result;
}
