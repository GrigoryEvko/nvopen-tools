// Function: sub_B6B020
// Address: 0xb6b020
//
__int64 __fastcall sub_B6B020(__int64 a1, int **a2, __int64 a3)
{
  int **v3; // r13
  __int64 v5; // rax
  __int64 *v6; // r14
  __int64 *v7; // r15
  unsigned int v8; // r12d
  int v10; // r13d
  __int64 v11; // r15
  unsigned int v12; // r14d
  unsigned int v13; // [rsp+4h] [rbp-7Ch]
  _BYTE *v14; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h]
  _BYTE v16[96]; // [rsp+20h] [rbp-60h] BYREF

  v3 = a2;
  v14 = v16;
  v15 = 0x200000000LL;
  if ( sub_B5F970(**(_QWORD **)(a1 + 16), a2, a3, (__int64)&v14, 0) )
  {
LABEL_16:
    v8 = 1;
    goto LABEL_7;
  }
  v13 = v15;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = (__int64 *)(v5 + 8LL * *(unsigned int *)(a1 + 12));
  if ( v6 != (__int64 *)(v5 + 8) )
  {
    v7 = (__int64 *)(v5 + 8);
    while ( 1 )
    {
      a2 = v3;
      if ( sub_B5F970(*v7, v3, a3, (__int64)&v14, 0) )
        goto LABEL_6;
      if ( v6 == ++v7 )
      {
        v10 = v15;
        goto LABEL_11;
      }
    }
  }
  v10 = v15;
LABEL_11:
  v11 = 0;
  v12 = 0;
  if ( v10 )
  {
    while ( 1 )
    {
      a2 = (int **)&v14[v11 + 8];
      if ( sub_B5F970(*(_QWORD *)&v14[v11], a2, a3, (__int64)&v14, 1u) )
        break;
      ++v12;
      v11 += 24;
      if ( v10 == v12 )
        goto LABEL_17;
    }
    if ( v13 <= v12 )
    {
LABEL_6:
      v8 = 2;
      goto LABEL_7;
    }
    goto LABEL_16;
  }
LABEL_17:
  v8 = 0;
LABEL_7:
  if ( v14 != v16 )
    _libc_free(v14, a2);
  return v8;
}
