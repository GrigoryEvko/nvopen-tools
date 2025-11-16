// Function: sub_C44830
// Address: 0xc44830
//
__int64 __fastcall sub_C44830(__int64 a1, _DWORD *a2, unsigned int a3)
{
  _DWORD *v3; // r13
  int v6; // edx
  const void *v7; // rsi
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  unsigned __int64 v11; // r15
  _QWORD *v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rcx
  unsigned __int64 v17; // [rsp+0h] [rbp-40h]
  char v18; // [rsp+Ch] [rbp-34h]

  v3 = a2;
  v6 = a2[2];
  if ( a3 > 0x40 )
  {
    if ( a3 == v6 )
    {
      *(_DWORD *)(a1 + 8) = a3;
      sub_C43780(a1, (const void **)a2);
    }
    else
    {
      v11 = ((unsigned __int64)a3 + 63) >> 6;
      v12 = (_QWORD *)sub_2207820(8 * v11);
      v13 = v12;
      v14 = (unsigned int)a2[2];
      if ( (unsigned int)v14 > 0x40 )
        a2 = *(_DWORD **)a2;
      v17 = (unsigned __int64)(v14 + 63) >> 6;
      v18 = v14;
      memcpy(v12, a2, 8 * v17);
      v13[(unsigned int)(v17 - 1)] = (__int64)(v13[(unsigned int)(v17 - 1)] << -v18) >> -v18;
      v15 = (unsigned int)v3[2];
      if ( (unsigned int)v15 <= 0x40 )
        v16 = *(_QWORD *)v3;
      else
        v16 = *(_QWORD *)(*(_QWORD *)v3 + 8LL * ((unsigned int)(v15 - 1) >> 6));
      memset(
        &v13[(unsigned __int64)(v15 + 63) >> 6],
        -((v16 & (1LL << ((unsigned __int8)v15 - 1))) != 0),
        8 * ((unsigned int)v11 - (unsigned int)((unsigned __int64)(v15 + 63) >> 6)));
      *(_DWORD *)(a1 + 8) = a3;
      *(_QWORD *)a1 = v13;
      v13[(unsigned int)(v11 - 1)] &= 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
    }
  }
  else
  {
    v7 = *(const void **)a2;
    v8 = 0;
    if ( v6 )
      v8 = (__int64)((_QWORD)v7 << (64 - (unsigned __int8)v6)) >> (64 - (unsigned __int8)v6);
    *(_DWORD *)(a1 + 8) = a3;
    v9 = v8 & (0xFFFFFFFFFFFFFFFFLL >> -(char)a3);
    if ( !a3 )
      v9 = 0;
    *(_QWORD *)a1 = v9;
  }
  return a1;
}
