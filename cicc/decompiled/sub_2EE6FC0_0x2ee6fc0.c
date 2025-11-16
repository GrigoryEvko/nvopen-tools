// Function: sub_2EE6FC0
// Address: 0x2ee6fc0
//
__int64 __fastcall sub_2EE6FC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _BYTE *v3; // r13
  _BYTE *v4; // r12
  _BYTE *v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r13
  int v9; // r15d
  _BYTE *v10; // r15
  __int64 i; // [rsp+18h] [rbp-38h]

  result = *(_QWORD *)(a2 + 56);
  for ( i = result; a2 + 48 != result; i = result )
  {
    v3 = *(_BYTE **)(i + 32);
    v4 = &v3[40 * (*(_DWORD *)(i + 40) & 0xFFFFFF)];
    if ( v3 != v4 )
    {
      while ( 1 )
      {
        v5 = v3;
        if ( sub_2DADC00(v3) )
          break;
        v3 += 40;
        if ( v4 == v3 )
          goto LABEL_15;
      }
      if ( v4 != v3 )
      {
        v8 = *(unsigned int *)(a1 + 8);
        do
        {
          v9 = *((_DWORD *)v5 + 2);
          if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, (const void *)(a1 + 16), v8 + 1, 4u, v6, v7);
            v8 = *(unsigned int *)(a1 + 8);
          }
          *(_DWORD *)(*(_QWORD *)a1 + 4 * v8) = v9;
          v8 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v8;
          if ( v5 + 40 == v4 )
            break;
          v10 = v5 + 40;
          while ( 1 )
          {
            v5 = v10;
            if ( sub_2DADC00(v10) )
              break;
            v10 += 40;
            if ( v4 == v10 )
              goto LABEL_15;
          }
        }
        while ( v4 != v10 );
      }
    }
LABEL_15:
    result = *(_QWORD *)(i + 8);
  }
  return result;
}
