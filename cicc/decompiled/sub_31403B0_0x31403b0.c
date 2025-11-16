// Function: sub_31403B0
// Address: 0x31403b0
//
__int64 __fastcall sub_31403B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 i; // rbx
  const void *v6; // r14
  size_t v7; // r12
  int v8; // eax
  unsigned int v9; // r15d
  __int64 *v10; // r9
  __int64 v11; // rax
  __int64 *v12; // r9
  _QWORD *v13; // rcx
  _QWORD *v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 *v16; // [rsp+18h] [rbp-38h]

  v3 = 16 * a3;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  result = a2 + v3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v15 = a2 + v3;
  if ( a2 != a2 + v3 )
  {
    for ( i = a2; v15 != i; i += 16 )
    {
      while ( 1 )
      {
        v6 = *(const void **)i;
        v7 = *(_QWORD *)(i + 8);
        v8 = sub_C92610();
        v9 = sub_C92740(a1, v6, v7, v8);
        v10 = (__int64 *)(*(_QWORD *)a1 + 8LL * v9);
        result = *v10;
        if ( *v10 )
          break;
LABEL_7:
        v16 = v10;
        v11 = sub_C7D670(v7 + 9, 8);
        v12 = v16;
        v13 = (_QWORD *)v11;
        if ( v7 )
        {
          v14 = (_QWORD *)v11;
          memcpy((void *)(v11 + 8), v6, v7);
          v12 = v16;
          v13 = v14;
        }
        *((_BYTE *)v13 + v7 + 8) = 0;
        i += 16;
        *v13 = v7;
        *v12 = (__int64)v13;
        ++*(_DWORD *)(a1 + 12);
        result = sub_C929D0((__int64 *)a1, v9);
        if ( v15 == i )
          return result;
      }
      if ( result == -8 )
      {
        --*(_DWORD *)(a1 + 16);
        goto LABEL_7;
      }
    }
  }
  return result;
}
