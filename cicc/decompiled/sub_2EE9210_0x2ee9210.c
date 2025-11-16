// Function: sub_2EE9210
// Address: 0x2ee9210
//
__int64 __fastcall sub_2EE9210(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 *v4; // rcx
  __int64 v5; // r8
  _BYTE *v7; // rdi
  unsigned int v8; // r12d
  __int64 v9; // rsi
  __int64 v10; // r8
  unsigned int v11; // r12d
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r9
  int v15; // eax
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // eax
  int v20; // eax
  int v21; // r10d
  _QWORD v22[2]; // [rsp+0h] [rbp-40h] BYREF
  _BYTE v23[48]; // [rsp+10h] [rbp-30h] BYREF

  v3 = *a1;
  v4 = *(__int64 **)(*a1 + 440LL);
  v5 = *(_QWORD *)(*(_QWORD *)(*v4 + 96)
                 + 0xFFFFFFFDD1745D18LL * (unsigned int)((__int64)(a1[1] - *(_QWORD *)(*a1 + 8LL)) >> 3));
  v22[0] = v23;
  v22[1] = 0x100000000LL;
  if ( v5 )
  {
    sub_2EE7A10(a2, (__int64)v22, v5, v4[3]);
    v7 = (_BYTE *)v22[0];
    v3 = *a1;
  }
  else
  {
    v7 = v23;
  }
  v8 = *(_DWORD *)(v3 + 400);
  v9 = *(_QWORD *)v7;
  v10 = *(_QWORD *)(v3 + 384);
  if ( v8 )
  {
    v11 = v8 - 1;
    v12 = v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v9 == *v13 )
    {
LABEL_5:
      v8 = *((_DWORD *)v13 + 2);
    }
    else
    {
      v20 = 1;
      while ( v14 != -4096 )
      {
        v21 = v20 + 1;
        v12 = v11 & (v20 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
          goto LABEL_5;
        v20 = v21;
      }
      v8 = 0;
    }
  }
  v15 = *(unsigned __int16 *)(v9 + 68);
  if ( (_WORD)v15 )
  {
    v16 = (unsigned int)(v15 - 9);
    if ( (unsigned __int16)v16 > 0x3Bu || (v17 = 0x800000000000C09LL, !_bittest64(&v17, v16)) )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v9 + 16) + 24LL) & 0x10) == 0 )
      {
        v18 = sub_2FF8170(*(_QWORD *)(v3 + 440) + 40LL, v9, *((unsigned int *)v7 + 2), a2, *((unsigned int *)v7 + 3));
        v7 = (_BYTE *)v22[0];
        v8 += v18;
      }
    }
  }
  if ( v7 != v23 )
    _libc_free((unsigned __int64)v7);
  return v8;
}
