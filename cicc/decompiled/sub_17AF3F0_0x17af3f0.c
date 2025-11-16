// Function: sub_17AF3F0
// Address: 0x17af3f0
//
__int64 __fastcall sub_17AF3F0(
        __int64 *a1,
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
  __int64 v10; // r15
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // r14
  _QWORD *v15; // rax
  _QWORD *v16; // r8
  _QWORD *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rbx
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  int v24; // eax
  void *s2; // [rsp+0h] [rbp-50h]
  __int64 n; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  _QWORD *v28; // [rsp+10h] [rbp-40h]
  int v29; // [rsp+1Ch] [rbp-34h]

  v10 = *(_QWORD *)(a2 + 8);
  if ( v10 && !*(_QWORD *)(v10 + 8) )
  {
    v27 = a2;
    v29 = 0;
    v12 = a2;
    v13 = *(_QWORD *)(a2 + 8);
    s2 = *(void **)(a2 + 56);
    v14 = *(unsigned int *)(a2 + 64);
    n = 4 * v14;
    while ( 1 )
    {
      v15 = sub_1648700(v13);
      v16 = v15;
      if ( *((_BYTE *)v15 + 16) != 87 )
        break;
      v18 = (*((_BYTE *)v15 + 23) & 0x40) != 0 ? (_QWORD *)*(v15 - 1) : &v15[-3 * (*((_DWORD *)v15 + 5) & 0xFFFFFFF)];
      if ( *v18 != v27 )
        break;
      if ( (_DWORD)v14 == *((_DWORD *)v16 + 16) )
      {
        if ( !n || (v28 = v16, v24 = memcmp((const void *)v16[7], s2, n), v16 = v28, !v24) )
        {
          v19 = *(_QWORD *)(a2 - 48);
          v20 = *a1;
          do
          {
            v21 = sub_1648700(v10);
            sub_170B990(v20, (__int64)v21);
            v10 = *(_QWORD *)(v10 + 8);
          }
          while ( v10 );
          if ( a2 == v19 )
            v19 = sub_1599EF0(*(__int64 ***)a2);
          sub_164D160(a2, v19, a3, a4, a5, a6, v22, v23, a9, a10);
          return v12;
        }
      }
      v13 = v16[1];
      ++v29;
      if ( !v13 || *(_QWORD *)(v13 + 8) || v29 == 10 )
        return 0;
      v27 = (__int64)v16;
    }
  }
  return 0;
}
