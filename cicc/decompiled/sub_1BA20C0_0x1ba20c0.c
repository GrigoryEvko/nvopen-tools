// Function: sub_1BA20C0
// Address: 0x1ba20c0
//
void __fastcall sub_1BA20C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // rbx
  _BYTE *v8; // rdi
  __int64 v9; // r12
  unsigned int v10; // ebx
  __int64 *v11; // r14
  unsigned int v12; // edx
  _BYTE *v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]
  _BYTE s[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 48) )
  {
    v7 = *(unsigned int *)(a2 + 4);
    v8 = s;
    v13 = s;
    v14 = 0x200000000LL;
    if ( (unsigned int)v7 > 2 )
    {
      sub_16CD150((__int64)&v13, s, v7, 8, a5, a6);
      v8 = v13;
    }
    LODWORD(v14) = v7;
    if ( 8 * v7 )
      memset(v8, 0, 8 * v7);
    v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 40LL)
                   + 8LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 48) + 48LL) - 1));
    if ( *(_DWORD *)(a2 + 4) )
    {
      v10 = 0;
      do
      {
        v11 = (__int64 *)&v13[8 * v10];
        v12 = v10++;
        *v11 = sub_1BA16F0(a2, v9, v12);
      }
      while ( *(_DWORD *)(a2 + 4) > v10 );
    }
    sub_1B9EC50(*(_QWORD *)(a2 + 224), *(_QWORD *)(a1 + 40), (__int64)&v13);
    if ( v13 != s )
      _libc_free((unsigned __int64)v13);
  }
  else
  {
    sub_1B9EC50(*(_QWORD *)(a2 + 224), *(_QWORD *)(a1 + 40), 0);
  }
}
