// Function: sub_39A0610
// Address: 0x39a0610
//
_QWORD *__fastcall sub_39A0610(__int64 a1, _QWORD *a2)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r14
  _QWORD *result; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r15
  _QWORD *v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rdi
  _QWORD *v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rsi
  unsigned __int64 v17; // rdi

  v4 = *(unsigned int *)(a1 + 176);
  v5 = *(unsigned int *)(a1 + 180);
  if ( (unsigned int)v4 >= (unsigned int)v5 )
  {
    v8 = ((((unsigned __int64)(v5 + 2) >> 1) | (v5 + 2)) >> 2) | ((unsigned __int64)(v5 + 2) >> 1) | (v5 + 2);
    v9 = (v8 >> 4) | v8;
    v10 = ((v9 >> 8) | v9 | (((v9 >> 8) | v9) >> 16) | (((v9 >> 8) | v9) >> 32)) + 1;
    if ( v10 > 0xFFFFFFFF )
      v10 = 0xFFFFFFFFLL;
    v6 = malloc(8 * v10);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v4 = *(unsigned int *)(a1 + 176);
    }
    v11 = *(_QWORD **)(a1 + 168);
    v12 = v4;
    v13 = (__int64)&v11[v12];
    if ( v11 == &v11[v12] )
    {
      v16 = (__int64)&v11[v12];
    }
    else
    {
      v14 = (_QWORD *)(v6 + v12 * 8);
      v15 = (_QWORD *)v6;
      do
      {
        if ( v15 )
        {
          *v15 = *v11;
          *v11 = 0;
        }
        ++v15;
        ++v11;
      }
      while ( v15 != v14 );
      v13 = *(_QWORD *)(a1 + 168);
      v16 = v13 + 8LL * *(unsigned int *)(a1 + 176);
    }
    sub_39A0390(v13, v16);
    v17 = *(_QWORD *)(a1 + 168);
    if ( v17 != a1 + 184 )
      _libc_free(v17);
    *(_QWORD *)(a1 + 168) = v6;
    LODWORD(v4) = *(_DWORD *)(a1 + 176);
    *(_DWORD *)(a1 + 180) = v10;
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 168);
  }
  result = (_QWORD *)(v6 + 8LL * (unsigned int)v4);
  if ( result )
  {
    *result = *a2;
    *a2 = 0;
    LODWORD(v4) = *(_DWORD *)(a1 + 176);
  }
  *(_DWORD *)(a1 + 176) = v4 + 1;
  return result;
}
