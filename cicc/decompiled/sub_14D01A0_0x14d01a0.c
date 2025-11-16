// Function: sub_14D01A0
// Address: 0x14d01a0
//
__int64 __fastcall sub_14D01A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v5; // r12
  __int64 v7; // rax
  __int64 *v8; // r14
  char v9; // dl
  __int64 v10; // r13
  __int64 *v11; // rsi
  unsigned int v12; // edi
  _QWORD *v13; // rcx
  __int64 v14; // [rsp-40h] [rbp-40h]

  result = (unsigned int)*(unsigned __int8 *)(a1 + 16) - 17;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 16) - 17) > 6u )
  {
    v5 = (__int64 *)a1;
    v7 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v8 = (__int64 *)(a1 - v7 * 8);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v8 = *(__int64 **)(a1 - 8);
      v5 = &v8[v7];
    }
    result = a3 + 16;
    v14 = a3 + 16;
    if ( v5 != v8 )
    {
      while ( 1 )
      {
        v10 = *v8;
        result = *(_QWORD *)(a2 + 8);
        if ( *(_QWORD *)(a2 + 16) != result )
          goto LABEL_6;
        v11 = (__int64 *)(result + 8LL * *(unsigned int *)(a2 + 28));
        v12 = *(_DWORD *)(a2 + 28);
        if ( (__int64 *)result != v11 )
        {
          v13 = 0;
          while ( v10 != *(_QWORD *)result )
          {
            if ( *(_QWORD *)result == -2 )
              v13 = (_QWORD *)result;
            result += 8;
            if ( v11 == (__int64 *)result )
            {
              if ( !v13 )
                goto LABEL_22;
              *v13 = v10;
              --*(_DWORD *)(a2 + 32);
              ++*(_QWORD *)a2;
              goto LABEL_17;
            }
          }
          goto LABEL_7;
        }
LABEL_22:
        if ( v12 < *(_DWORD *)(a2 + 24) )
        {
          *(_DWORD *)(a2 + 28) = v12 + 1;
          *v11 = v10;
          ++*(_QWORD *)a2;
LABEL_17:
          result = sub_14AF470(v10, 0, 0, 0);
          if ( !(_BYTE)result )
            goto LABEL_7;
          result = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)result >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, v14, 0, 8);
            result = *(unsigned int *)(a3 + 8);
          }
          v8 += 3;
          *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v10;
          ++*(_DWORD *)(a3 + 8);
          if ( v5 == v8 )
            return result;
        }
        else
        {
LABEL_6:
          result = sub_16CCBA0(a2, *v8);
          if ( v9 )
            goto LABEL_17;
LABEL_7:
          v8 += 3;
          if ( v5 == v8 )
            return result;
        }
      }
    }
  }
  return result;
}
