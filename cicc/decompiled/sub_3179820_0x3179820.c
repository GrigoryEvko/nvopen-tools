// Function: sub_3179820
// Address: 0x3179820
//
unsigned __int64 __fastcall sub_3179820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 result; // rax
  bool v10; // al
  __int64 *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned int v15; // edx
  __int64 *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  int v21; // esi
  int v22; // r9d
  unsigned __int64 v23; // [rsp-80h] [rbp-80h]
  _BYTE *v24; // [rsp-78h] [rbp-78h] BYREF
  __int64 v25; // [rsp-70h] [rbp-70h]
  _BYTE v26[104]; // [rsp-68h] [rbp-68h] BYREF

  v5 = *(_QWORD *)(a1 + 240);
  v6 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)v5 != v6 )
    return 0;
  v10 = sub_AD7A80(*(_BYTE **)(v5 + 8), a2, a3, v6, a5);
  v11 = *(__int64 **)(a1 + 56);
  v24 = v26;
  v12 = *(_QWORD *)(a2 - 32LL * v10 - 32);
  v25 = 0x600000000LL;
  if ( (unsigned __int8)sub_2A64220(v11, v12) )
  {
    v13 = *(unsigned int *)(a1 + 120);
    v14 = *(_QWORD *)(a1 + 104);
    if ( (_DWORD)v13 )
    {
      v15 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v16 = (__int64 *)(v14 + 8LL * v15);
      v17 = *v16;
      if ( v12 == *v16 )
      {
LABEL_9:
        if ( v16 != (__int64 *)(v14 + 8 * v13) )
          goto LABEL_4;
      }
      else
      {
        v21 = 1;
        while ( v17 != -4096 )
        {
          v22 = v21 + 1;
          v15 = (v13 - 1) & (v21 + v15);
          v16 = (__int64 *)(v14 + 8LL * v15);
          v17 = *v16;
          if ( v12 == *v16 )
            goto LABEL_9;
          v21 = v22;
        }
      }
    }
    if ( (unsigned __int8)sub_3175050(a1, *(_QWORD *)(a2 + 40), v12) )
    {
      v20 = (unsigned int)v25;
      if ( (unsigned __int64)(unsigned int)v25 + 1 > HIDWORD(v25) )
      {
        sub_C8D5F0((__int64)&v24, v26, (unsigned int)v25 + 1LL, 8u, v18, v19);
        v20 = (unsigned int)v25;
      }
      *(_QWORD *)&v24[8 * v20] = v12;
      LODWORD(v25) = v25 + 1;
    }
  }
LABEL_4:
  result = sub_3178E50(a1, (__int64)&v24);
  if ( v24 != v26 )
  {
    v23 = result;
    _libc_free((unsigned __int64)v24);
    return v23;
  }
  return result;
}
