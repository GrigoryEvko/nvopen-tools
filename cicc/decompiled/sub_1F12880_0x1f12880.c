// Function: sub_1F12880
// Address: 0x1f12880
//
_QWORD *__fastcall sub_1F12880(_QWORD *a1, _DWORD *a2, __int64 a3, char a4, int a5, int a6)
{
  _QWORD *result; // rax
  __int64 *v7; // rdx
  _DWORD *v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rcx
  unsigned int v12; // r12d
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  __int64 v19; // rsi
  _DWORD *v20; // [rsp+8h] [rbp-48h]
  __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  result = &a2[a3];
  v7 = v21;
  v20 = result;
  if ( result != (_QWORD *)a2 )
  {
    v9 = a2;
    do
    {
      v18 = (unsigned int)*v9;
      v19 = *(_QWORD *)(a1[47] + 8 * v18);
      v21[0] = v19;
      if ( a4 )
      {
        sub_16AF570(v21, v19);
        LODWORD(v18) = *v9;
      }
      ++v9;
      v10 = *(_QWORD *)(a1[30] + 240LL);
      v11 = (unsigned int)(2 * v18);
      v12 = *(_DWORD *)(v10 + 4 * v11);
      v13 = *(unsigned int *)(v10 + 4LL * (unsigned int)(v11 + 1));
      sub_1F12210((__int64)a1, v12, (__int64)v7, v11, a5, a6);
      sub_1F12210((__int64)a1, v13, v14, v15, v16, v17);
      sub_16AF570((_QWORD *)(a1[33] + 112LL * v12), v21[0]);
      result = sub_16AF570((_QWORD *)(a1[33] + 112 * v13), v21[0]);
    }
    while ( v9 != v20 );
  }
  return result;
}
