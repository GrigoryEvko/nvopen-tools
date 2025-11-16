// Function: sub_28CED90
// Address: 0x28ced90
//
__int64 *__fastcall sub_28CED90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 *a5, __int64 a6)
{
  unsigned __int64 v8; // rbx
  _QWORD *v9; // r8
  unsigned int v10; // eax
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r10
  __int64 v19; // rbx
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  unsigned __int64 v22; // r9
  unsigned int v23; // r15d
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-50h]
  unsigned __int64 v30; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v31; // [rsp+18h] [rbp-38h] BYREF

  if ( !a5 )
    goto LABEL_17;
  if ( *a5 <= 0x15u )
  {
    v8 = *(unsigned int *)(a3 + 32);
    v9 = *(_QWORD **)(a3 + 24);
    v10 = 0;
    if ( *(_DWORD *)(a3 + 32) )
    {
      if ( --v8 )
      {
        _BitScanReverse64(&v8, v8);
        v10 = 64 - (v8 ^ 0x3F);
        v8 = 8LL * (int)v10;
      }
    }
    v11 = *(unsigned int *)(a2 + 176);
    if ( (unsigned int)v11 <= v10 )
    {
      v22 = v10 + 1;
      v23 = v10 + 1;
      if ( v22 != v11 )
      {
        if ( v22 >= v11 )
        {
          if ( v22 > *(unsigned int *)(a2 + 180) )
          {
            v28 = v9;
            v30 = v10 + 1;
            sub_C8D5F0(a2 + 168, (const void *)(a2 + 184), v22, 8u, (__int64)v9, v22);
            v11 = *(unsigned int *)(a2 + 176);
            v9 = v28;
            v22 = v30;
          }
          v12 = *(_QWORD *)(a2 + 168);
          v24 = (_QWORD *)(v12 + 8 * v11);
          v25 = (_QWORD *)(v12 + 8 * v22);
          if ( v24 != v25 )
          {
            do
            {
              if ( v24 )
                *v24 = 0;
              ++v24;
            }
            while ( v25 != v24 );
            v12 = *(_QWORD *)(a2 + 168);
          }
          *(_DWORD *)(a2 + 176) = v23;
          goto LABEL_8;
        }
        *(_DWORD *)(a2 + 176) = v22;
      }
    }
    v12 = *(_QWORD *)(a2 + 168);
LABEL_8:
    *v9 = *(_QWORD *)(v12 + v8);
    *(_QWORD *)(*(_QWORD *)(a2 + 168) + v8) = v9;
    v13 = sub_28CECC0(a2, a5);
    a1[1] = 0;
    *a1 = v13;
    a1[2] = 0;
    return a1;
  }
  if ( *a5 != 22 )
  {
    v31 = a5;
    v15 = sub_28C7580(a2 + 1432, (__int64 *)&v31);
    if ( v15 )
    {
      v19 = v15[1];
      if ( v19 )
      {
        v20 = *(unsigned __int8 **)(v19 + 8);
        if ( v20 && (unsigned __int8 *)a4 != v20 )
        {
          v21 = sub_28CED20(a2, v20);
          a1[1] = (__int64)a5;
          *a1 = v21;
          a1[2] = 0;
          return a1;
        }
        if ( *(_QWORD *)(v19 + 56) )
        {
          sub_28C79C0(*(_QWORD **)(v18 + 24), *(_DWORD *)(v18 + 32), a2 + 168, a4, v16, v17);
          v27 = *(_QWORD *)(v19 + 56);
          a1[1] = (__int64)a5;
          a1[2] = 0;
          *a1 = v27;
          return a1;
        }
      }
    }
LABEL_17:
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
    return a1;
  }
  sub_28C79C0(*(_QWORD **)(a3 + 24), *(_DWORD *)(a3 + 32), a2 + 168, a4, (__int64)a5, a6);
  v26 = sub_A777F0(0x20u, (__int64 *)(a2 + 72));
  if ( v26 )
  {
    *(_QWORD *)(v26 + 16) = 0;
    *(_QWORD *)(v26 + 8) = 0xFFFFFFFD00000002LL;
    *(_QWORD *)(v26 + 24) = a5;
    *(_QWORD *)v26 = &unk_4A21968;
  }
  *(_DWORD *)(v26 + 12) = *a5;
  *a1 = v26;
  a1[1] = 0;
  a1[2] = 0;
  return a1;
}
