// Function: sub_3507C70
// Address: 0x3507c70
//
__int64 __fastcall sub_3507C70(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 result; // rax
  const void *v10; // rsi
  _WORD *v12; // rbx
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rax
  _WORD *v16; // rdx
  __int64 v17; // rax
  int v18; // eax

  v7 = a1[1];
  result = 2LL * a1[2];
  if ( result )
  {
    v10 = (const void *)(a3 + 16);
    v12 = (_WORD *)a1[1];
    do
    {
      while ( 1 )
      {
        v18 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 4 * ((unsigned __int64)(unsigned __int16)*v12 >> 5));
        if ( !_bittest(&v18, (unsigned __int16)*v12) )
          break;
        ++v12;
        result = v7 + 2LL * a1[2];
        if ( v12 == (_WORD *)result )
          return result;
      }
      if ( a3 )
      {
        v13 = *(unsigned int *)(a3 + 8);
        LOWORD(v6) = *v12;
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, v10, v13 + 1, 0x10u, a5, a6);
          v13 = *(unsigned int *)(a3 + 8);
        }
        v14 = (_QWORD *)(*(_QWORD *)a3 + 16 * v13);
        *v14 = v6;
        v14[1] = a2;
        ++*(_DWORD *)(a3 + 8);
        v7 = a1[1];
      }
      v15 = a1[2];
      v16 = (_WORD *)(v7 + 2 * v15 - 2);
      if ( v16 != v12 )
      {
        *v12 = *v16;
        *(_BYTE *)(a1[6] + *(unsigned __int16 *)(a1[1] + 2LL * a1[2] - 2)) = ((__int64)v12 - a1[1]) >> 1;
        v15 = a1[2];
        v7 = a1[1];
      }
      v17 = v15 - 1;
      a1[2] = v17;
      result = v7 + 2 * v17;
    }
    while ( v12 != (_WORD *)result );
  }
  return result;
}
