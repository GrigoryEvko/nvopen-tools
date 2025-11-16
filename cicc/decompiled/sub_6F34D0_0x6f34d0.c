// Function: sub_6F34D0
// Address: 0x6f34d0
//
__int64 __fastcall sub_6F34D0(__int64 a1)
{
  __int64 v1; // rbx
  unsigned int v2; // r12d
  _QWORD *v3; // rax
  _QWORD *v4; // r13
  _QWORD *v5; // rsi
  __int64 *v6; // r14
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 *i; // rax
  __int64 v12; // rsi
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  v1 = *(unsigned int *)(a1 + 8);
  v2 = 2 * v1 + 1;
  v14 = v1 + 1;
  v3 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v1 + 2));
  v4 = v3;
  if ( 2 * (_DWORD)v1 != -2 )
  {
    v5 = &v3[2 * v2 + 2];
    do
    {
      if ( v3 )
        *v3 = 0;
      v3 += 2;
    }
    while ( v3 != v5 );
  }
  v6 = *(__int64 **)a1;
  if ( (_DWORD)v1 != -1 )
  {
    v7 = *(__int64 **)a1;
    v8 = (__int64)&v6[2 * v1 + 2];
    do
    {
      if ( *v7 )
      {
        v9 = v2 & (unsigned int)sub_72DB90();
        v10 = v9;
        for ( i = &v4[2 * v9]; *i; i = &v4[2 * v10] )
          v10 = v2 & (v10 + 1);
        v12 = *v7;
        *i = *v7;
        if ( v12 )
          i[1] = v7[1];
      }
      v7 += 2;
    }
    while ( (__int64 *)v8 != v7 );
  }
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 8) = v2;
  return sub_823A00(v6, 16LL * v14);
}
