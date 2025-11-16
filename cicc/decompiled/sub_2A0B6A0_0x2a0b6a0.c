// Function: sub_2A0B6A0
// Address: 0x2a0b6a0
//
bool __fastcall sub_2A0B6A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rbx
  bool result; // al
  __int64 v5; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 *v11; // r12
  unsigned __int64 v12; // r8
  bool v13; // [rsp+Fh] [rbp-51h]
  __int64 *v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h]
  _BYTE v16[64]; // [rsp+20h] [rbp-40h] BYREF

  v2 = sub_D47930(a1);
  v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v2 + 48 )
    goto LABEL_43;
  if ( !v3 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
LABEL_43:
    BUG();
  if ( *(_BYTE *)(v3 - 24) != 31 || (*(_DWORD *)(v3 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v5 = *(_QWORD *)(v3 - 88);
  if ( *(_BYTE *)(a1 + 84) )
  {
    v6 = *(_QWORD **)(a1 + 64);
    v7 = &v6[*(unsigned int *)(a1 + 76)];
    if ( v6 == v7 )
      goto LABEL_14;
    while ( v5 != *v6 )
    {
      if ( v7 == ++v6 )
        goto LABEL_14;
    }
LABEL_13:
    v5 = *(_QWORD *)(v3 - 56);
    goto LABEL_14;
  }
  a2 = *(_QWORD *)(v3 - 88);
  if ( sub_C8CA60(a1 + 56, a2) )
    goto LABEL_13;
LABEL_14:
  if ( !sub_AA5820(v5, a2) )
    return 0;
  v15 = 0x400000000LL;
  v14 = (__int64 *)v16;
  sub_D474A0(a1, (__int64)&v14);
  result = 0;
  if ( (_DWORD)v15 )
  {
    v8 = v14;
    v9 = &v14[(unsigned int)v15];
    v10 = (8LL * (unsigned int)v15) >> 3;
    if ( (8LL * (unsigned int)v15) >> 5 )
    {
      v11 = &v14[4 * ((8LL * (unsigned int)v15) >> 5)];
      while ( 1 )
      {
        if ( !sub_AA5820(*v8, (__int64)&v14) )
        {
          result = v8 != v9;
          goto LABEL_16;
        }
        if ( !sub_AA5820(v8[1], (__int64)&v14) )
        {
          result = v9 != v8 + 1;
          goto LABEL_16;
        }
        if ( !sub_AA5820(v8[2], (__int64)&v14) )
        {
          result = v9 != v8 + 2;
          goto LABEL_16;
        }
        if ( !sub_AA5820(v8[3], (__int64)&v14) )
          break;
        v8 += 4;
        if ( v11 == v8 )
        {
          v10 = v9 - v8;
          goto LABEL_29;
        }
      }
      result = v9 != v8 + 3;
      goto LABEL_16;
    }
LABEL_29:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
        {
          result = 0;
          goto LABEL_16;
        }
LABEL_40:
        v12 = sub_AA5820(*v8, (__int64)&v14);
        result = 0;
        if ( v12 )
          goto LABEL_16;
        goto LABEL_41;
      }
      if ( !sub_AA5820(*v8, (__int64)&v14) )
      {
LABEL_41:
        result = v9 != v8;
        goto LABEL_16;
      }
      ++v8;
    }
    if ( sub_AA5820(*v8, (__int64)&v14) )
    {
      ++v8;
      goto LABEL_40;
    }
    goto LABEL_41;
  }
LABEL_16:
  if ( v14 != (__int64 *)v16 )
  {
    v13 = result;
    _libc_free((unsigned __int64)v14);
    return v13;
  }
  return result;
}
