// Function: sub_D23CB0
// Address: 0xd23cb0
//
char __fastcall sub_D23CB0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rcx
  int v8; // eax
  int v9; // edi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r8
  int v14; // eax
  int v15; // r9d

  if ( a2 == a3 )
    return 1;
  v4 = **a1;
  v5 = sub_D23C40(v4, a2);
  v6 = *(_QWORD *)(v4 + 312);
  v7 = v5;
  v8 = *(_DWORD *)(v4 + 328);
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (__int64 *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a3 )
      return v7 == v11[1];
    v14 = 1;
    while ( v12 != -4096 )
    {
      v15 = v14 + 1;
      v10 = v9 & (v14 + v10);
      v11 = (__int64 *)(v6 + 16LL * v10);
      v12 = *v11;
      if ( a3 == *v11 )
        return v7 == v11[1];
      v14 = v15;
    }
  }
  return v7 == 0;
}
