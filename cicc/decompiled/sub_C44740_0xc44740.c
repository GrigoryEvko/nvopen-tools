// Function: sub_C44740
// Address: 0xc44740
//
__int64 __fastcall sub_C44740(__int64 a1, char **a2, unsigned int a3)
{
  __int64 *v3; // r13
  unsigned int v6; // eax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v10; // rax
  char *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  int v14; // ecx

  v3 = (__int64 *)a2;
  v6 = *((_DWORD *)a2 + 2);
  if ( a3 > 0x40 )
  {
    if ( a3 == v6 )
    {
      *(_DWORD *)(a1 + 8) = a3;
      sub_C43780(a1, (const void **)a2);
      return a1;
    }
    else
    {
      v10 = sub_2207820(8 * (((unsigned __int64)a3 + 63) >> 6));
      v11 = *a2;
      v12 = v10;
      v13 = 0;
      do
      {
        *(_QWORD *)(v12 + v13) = *(_QWORD *)&v11[v13];
        v13 += 8;
      }
      while ( v13 != 8LL * (a3 >> 6) );
      v14 = -a3 & 0x3F;
      if ( v14 )
        *(_QWORD *)(v12 + v13) = *(_QWORD *)&v11[v13] << v14 >> v14;
      *(_DWORD *)(a1 + 8) = a3;
      *(_QWORD *)a1 = v12;
      return a1;
    }
  }
  else
  {
    if ( v6 > 0x40 )
      v3 = (__int64 *)*a2;
    v7 = *v3;
    *(_DWORD *)(a1 + 8) = a3;
    v8 = (0xFFFFFFFFFFFFFFFFLL >> -(char)a3) & v7;
    if ( !a3 )
      v8 = 0;
    *(_QWORD *)a1 = v8;
    return a1;
  }
}
