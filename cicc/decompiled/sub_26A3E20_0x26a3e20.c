// Function: sub_26A3E20
// Address: 0x26a3e20
//
__int64 __fastcall sub_26A3E20(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rdx
  _QWORD *v5; // r12
  __int64 v7; // r13
  unsigned int *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // edx
  unsigned __int8 **v15; // rax
  unsigned __int8 *v16; // rcx
  int v17; // r11d
  unsigned __int8 **v18; // r10
  int v19; // edx
  unsigned __int8 **v20; // rax
  __int64 v21; // [rsp-70h] [rbp-70h]
  unsigned __int8 **v22; // [rsp-60h] [rbp-60h] BYREF
  unsigned __int8 *v23; // [rsp-58h] [rbp-58h] BYREF
  __int64 v24; // [rsp-50h] [rbp-50h]
  __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-40h] [rbp-40h]

  if ( (unsigned __int8)(*a2 - 34) > 0x33u )
    return 1;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, (unsigned int)*a2 - 34) )
    return 1;
  v5 = a2 + 72;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(unsigned int **)(a1 + 8);
  v21 = *(_QWORD *)a1;
  if ( !(unsigned __int8)sub_A747A0((_QWORD *)a2 + 9, "no_openmp", 9u)
    && !(unsigned __int8)sub_B49590((__int64)a2, "no_openmp", 9u)
    && !(unsigned __int8)sub_A747A0(v5, "no_openmp_routines", 0x12u)
    && !(unsigned __int8)sub_B49590((__int64)a2, "no_openmp_routines", 0x12u)
    && !(unsigned __int8)sub_A747A0(v5, "no_openmp_constructs", 0x14u)
    && !(unsigned __int8)sub_B49590((__int64)a2, "no_openmp_constructs", 0x14u) )
  {
    v9 = sub_26A3510(v7, v21, (unsigned __int64)a2, v8);
    v26 = v10;
    v25 = v9;
    if ( (_BYTE)v10 )
    {
      v11 = *(_QWORD *)(a1 + 24);
      v23 = a2;
      v24 = v9;
      v12 = *(_DWORD *)(v11 + 24);
      if ( !v12 )
      {
        v22 = 0;
        ++*(_QWORD *)v11;
        goto LABEL_24;
      }
      v13 = *(_QWORD *)(v11 + 8);
      v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = (unsigned __int8 **)(v13 + 16LL * v14);
      v16 = *v15;
      if ( a2 != *v15 )
      {
        v17 = 1;
        v18 = 0;
        while ( v16 != (unsigned __int8 *)-4096LL )
        {
          if ( v18 || v16 != (unsigned __int8 *)-8192LL )
            v15 = v18;
          v14 = (v12 - 1) & (v17 + v14);
          v16 = *(unsigned __int8 **)(v13 + 16LL * v14);
          if ( a2 == v16 )
            return 1;
          ++v17;
          v18 = v15;
          v15 = (unsigned __int8 **)(v13 + 16LL * v14);
        }
        if ( !v18 )
          v18 = v15;
        v19 = *(_DWORD *)(v11 + 16) + 1;
        v22 = v18;
        ++*(_QWORD *)v11;
        if ( 4 * v19 < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(v11 + 20) - v19 > v12 >> 3 )
          {
LABEL_20:
            v20 = v22;
            *(_DWORD *)(v11 + 16) = v19;
            if ( *v20 != (unsigned __int8 *)-4096LL )
              --*(_DWORD *)(v11 + 20);
            *v20 = v23;
            v20[1] = (unsigned __int8 *)v24;
            **(_DWORD **)(a1 + 32) = 0;
            return 1;
          }
LABEL_25:
          sub_2685CE0(v11, v12);
          sub_2677F80(v11, (__int64 *)&v23, &v22);
          v19 = *(_DWORD *)(v11 + 16) + 1;
          goto LABEL_20;
        }
LABEL_24:
        v12 *= 2;
        goto LABEL_25;
      }
    }
  }
  return 1;
}
