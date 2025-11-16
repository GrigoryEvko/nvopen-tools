// Function: sub_177D390
// Address: 0x177d390
//
void __fastcall sub_177D390(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 i; // r12
  int v11; // r8d
  int v12; // r9d
  _QWORD *v13; // r13
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 *v17; // r15
  __int64 v18; // rsi
  double v19; // xmm4_8
  double v20; // xmm5_8
  double v21; // xmm4_8
  double v22; // xmm5_8

  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v13 = sub_1648700(i);
    v14 = *((_BYTE *)v13 + 16);
    if ( v14 <= 0x17u )
      break;
    if ( v14 == 54 )
    {
      v16 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      v17 = *(__int64 **)a1;
      if ( *(_QWORD *)a1 != v16 )
      {
        do
        {
          v18 = *v17;
          if ( !sub_1776680(a1, *v17) )
            sub_177CC40(a1, v18, a3, a4, a5, a6, v19, v20, a9, a10);
          ++v17;
        }
        while ( (__int64 *)v16 != v17 );
      }
      if ( !sub_1776680(a1, (__int64)v13) )
        sub_177CC40(a1, (__int64)v13, a3, a4, a5, a6, v21, v22, a9, a10);
    }
    else
    {
      if ( v14 != 56 && v14 != 71 )
        return;
      v15 = *(unsigned int *)(a1 + 8);
      if ( (unsigned int)v15 >= *(_DWORD *)(a1 + 12) )
      {
        sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v11, v12);
        v15 = *(unsigned int *)(a1 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v15) = v13;
      ++*(_DWORD *)(a1 + 8);
      sub_177D390(a1, v13);
      --*(_DWORD *)(a1 + 8);
    }
  }
}
