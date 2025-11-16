// Function: sub_16A5B10
// Address: 0x16a5b10
//
__int64 __fastcall sub_16A5B10(__int64 a1, _DWORD *a2, unsigned int a3)
{
  unsigned __int64 v3; // r14
  _DWORD *v4; // r13
  __int64 v6; // rax
  int v7; // ecx
  unsigned __int64 v9; // r15
  void *v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  unsigned __int64 v15; // [rsp+0h] [rbp-40h]
  char v16; // [rsp+8h] [rbp-38h]
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
  v4 = a2;
  if ( a3 > 0x40 )
  {
    v9 = ((unsigned __int64)a3 + 63) >> 6;
    v10 = (void *)sub_2207820(8 * v9);
    v11 = (unsigned int)a2[2];
    if ( (unsigned int)v11 > 0x40 )
      a2 = *(_DWORD **)a2;
    v15 = (unsigned __int64)(v11 + 63) >> 6;
    v16 = v11;
    v12 = memcpy(v10, a2, 8 * v15);
    v12[(unsigned int)(v15 - 1)] = (__int64)(v12[(unsigned int)(v15 - 1)] << -v16) >> -v16;
    v13 = (unsigned int)v4[2];
    if ( (unsigned int)v13 <= 0x40 )
      v14 = *(_QWORD *)v4;
    else
      v14 = *(_QWORD *)(*(_QWORD *)v4 + 8LL * ((unsigned int)(v13 - 1) >> 6));
    v17 = v12;
    memset(
      &v12[(unsigned __int64)(v13 + 63) >> 6],
      -((v14 & (1LL << ((unsigned __int8)v13 - 1))) != 0),
      8 * ((unsigned int)v9 - (unsigned int)((unsigned __int64)(v13 + 63) >> 6)));
    v17[(unsigned int)(v9 - 1)] &= v3;
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v17;
  }
  else
  {
    v6 = *(_QWORD *)a2;
    v7 = 64 - a2[2];
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = (v6 << v7 >> v7) & v3;
  }
  return a1;
}
