// Function: sub_1E41E60
// Address: 0x1e41e60
//
__int64 __fastcall sub_1E41E60(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 *v6; // r14
  __int64 v7; // r12
  __int64 v8; // r15
  int v9; // r8d
  int v10; // r9d

  result = *a1;
  v4 = *(_QWORD *)*a1;
  if ( (v4 & 4) == 0 )
  {
    v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5 )
    {
      sub_14AD470(v5, a2, a3, 0, 6u);
      v6 = *(__int64 **)a2;
      result = *(unsigned int *)(a2 + 8);
      v7 = *(_QWORD *)a2 + 8 * result;
      if ( *(_QWORD *)a2 != v7 )
      {
        while ( 1 )
        {
          v8 = *v6;
          result = sub_134E860(*v6);
          if ( !(_BYTE)result )
            break;
          result = *(unsigned int *)(a2 + 8);
          if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
          {
            sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v9, v10);
            result = *(unsigned int *)(a2 + 8);
          }
          ++v6;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v8;
          ++*(_DWORD *)(a2 + 8);
          if ( (__int64 *)v7 == v6 )
            return result;
        }
        *(_DWORD *)(a2 + 8) = 0;
      }
    }
  }
  return result;
}
