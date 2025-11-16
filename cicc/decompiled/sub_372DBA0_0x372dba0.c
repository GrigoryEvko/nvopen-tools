// Function: sub_372DBA0
// Address: 0x372dba0
//
void __fastcall sub_372DBA0(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v6; // r14
  unsigned int *v7; // rdi
  unsigned int *v8; // rbx
  unsigned int *v9; // r15
  unsigned int v10; // esi
  int *v11; // rax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r13
  __int64 i; // [rsp+20h] [rbp-60h]
  unsigned int *v19; // [rsp+30h] [rbp-50h] BYREF
  __int64 v20; // [rsp+38h] [rbp-48h]
  _BYTE v21[64]; // [rsp+40h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 40);
  for ( i = v6 + 16LL * *(unsigned int *)(a2 + 48); i != v6; v6 += 16 )
  {
    v19 = (unsigned int *)v21;
    v20 = 0x400000000LL;
    sub_372D1A0(*(_OWORD *)v6, *(_DWORD *)(a2 + 32), a5, a4, a3, (__int64)&v19);
    v7 = v19;
    v8 = &v19[(unsigned int)v20];
    if ( v8 != v19 )
    {
      v9 = v19;
      do
      {
        v10 = *v9++;
        sub_372A170(a1, v10, *(_QWORD *)v6, *(_QWORD *)(v6 + 8));
      }
      while ( v8 != v9 );
      v7 = v19;
    }
    if ( v7 != (unsigned int *)v21 )
      _libc_free((unsigned __int64)v7);
  }
  v11 = sub_220F330((int *)a2, a1 + 1);
  v12 = *((_QWORD *)v11 + 5);
  v13 = (unsigned __int64)v11;
  if ( (int *)v12 != v11 + 14 )
    _libc_free(v12);
  j_j___libc_free_0(v13);
  --a1[5];
}
