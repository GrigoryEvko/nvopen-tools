// Function: sub_2EBE740
// Address: 0x2ebe740
//
__int64 __fastcall sub_2EBE740(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // esi
  unsigned int v8; // edx
  unsigned __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // r14
  unsigned __int64 v12; // r15
  _QWORD *v13; // rax
  unsigned __int64 v14; // rdx

  v6 = a2 & 0x7FFFFFFF;
  v8 = v6 + 1;
  v9 = *(unsigned int *)(a1 + 464);
  if ( v6 + 1 > (unsigned int)v9 )
  {
    v11 = *(_QWORD *)(a1 + 472);
    if ( v8 != v9 )
    {
      if ( v8 >= v9 )
      {
        v12 = v8 - v9;
        if ( v8 > (unsigned __int64)*(unsigned int *)(a1 + 468) )
        {
          sub_C8D5F0(a1 + 456, (const void *)(a1 + 472), v8, 8u, v8, a6);
          v9 = *(unsigned int *)(a1 + 464);
        }
        v13 = (_QWORD *)(*(_QWORD *)(a1 + 456) + 8 * v9);
        v14 = v12;
        do
        {
          if ( v13 )
            *v13 = v11;
          ++v13;
          --v14;
        }
        while ( v14 );
        *(_DWORD *)(a1 + 464) += v12;
      }
      else
      {
        *(_DWORD *)(a1 + 464) = v8;
      }
    }
  }
  result = *(_QWORD *)(a1 + 456);
  *(_QWORD *)(result + 8LL * v6) = a3;
  return result;
}
