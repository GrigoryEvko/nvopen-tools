// Function: sub_1C8D510
// Address: 0x1c8d510
//
__int64 __fastcall sub_1C8D510(
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
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rsi
  unsigned int v24; // r12d
  int v26; // [rsp+Ch] [rbp-94h]
  unsigned __int8 v28; // [rsp+37h] [rbp-69h] BYREF
  int v29; // [rsp+38h] [rbp-68h] BYREF
  char v30; // [rsp+3Ch] [rbp-64h]
  _QWORD v31[2]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v32[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v33[8]; // [rsp+60h] [rbp-40h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_20:
    JUMPOUT(0x41FFE4);
  while ( *(_UNKNOWN **)v11 != &unk_4FB9E2C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_20;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4FB9E2C);
  v28 = 0;
  v16 = *(_QWORD *)(a2 + 32);
  v26 = *(_DWORD *)(v13 + 164);
  if ( a2 + 24 == v16 )
  {
    v24 = *(unsigned __int8 *)(a1 + 153);
    if ( !(_BYTE)v24 )
      return v24;
    v24 = 0;
  }
  else
  {
    do
    {
      v17 = v16;
      v16 = *(_QWORD *)(v16 + 8);
      v18 = *(_QWORD *)(v17 + 24);
      v19 = v17 + 16;
      while ( v19 != v18 )
      {
        while ( 1 )
        {
          v20 = v18;
          v18 = *(_QWORD *)(v18 + 8);
          v21 = *(_QWORD *)(v20 + 24);
          v22 = v20 + 16;
          if ( v20 + 16 == v21 )
            break;
          do
          {
            v23 = v21;
            v21 = *(_QWORD *)(v21 + 8);
            sub_1C8C170(&v28, v23 - 24, a3, a4, a5, a6, v14, v15, a9, a10);
          }
          while ( v22 != v21 );
          if ( v19 == v18 )
            goto LABEL_11;
        }
      }
LABEL_11:
      ;
    }
    while ( a2 + 24 != v16 );
    v24 = v28;
    if ( !*(_BYTE *)(a1 + 153) )
    {
      if ( !v28 )
        return v24;
      goto LABEL_18;
    }
  }
  LOBYTE(v32[0]) = 0;
  sub_1C8B580(v32, a2, a3, a4, a5, a6, v14, v15, a9, a10);
  LOBYTE(v24) = LOBYTE(v32[0]) | v24;
  if ( !(_BYTE)v24 )
    return v24;
LABEL_18:
  v31[0] = &unk_42D2EE0;
  v33[0] = v31;
  v32[1] = 0x100000001LL;
  v31[1] = 49900;
  v32[0] = (unsigned __int64)v33;
  v30 = 0;
  v29 = 10 * v26;
  sub_1CCF3C0(a2, (unsigned int)v33, 1, (unsigned int)&v29, 0, 0, 0);
  if ( (_QWORD *)v32[0] != v33 )
    _libc_free(v32[0]);
  return v24;
}
