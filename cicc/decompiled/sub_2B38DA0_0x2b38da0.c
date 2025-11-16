// Function: sub_2B38DA0
// Address: 0x2b38da0
//
void __fastcall sub_2B38DA0(unsigned int *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // rdx
  __int64 v8; // r14
  int v9; // r15d
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 *v14; // rax
  unsigned __int64 v15; // [rsp+0h] [rbp-80h]
  __int64 *v16; // [rsp+10h] [rbp-70h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h]
  _QWORD v18[12]; // [rsp+20h] [rbp-60h] BYREF

  v3 = sub_ACADE0(*(__int64 ***)(**(_QWORD **)a1 + 8LL));
  v7 = a1[2];
  v16 = v18;
  v8 = v3;
  v17 = 0x600000000LL;
  v9 = v7;
  if ( (unsigned int)v7 > 6 )
  {
    v15 = v7;
    sub_C8D5F0((__int64)&v16, v18, v7, 8u, v5, v6);
    v14 = v16;
    v7 = (unsigned __int64)&v16[v15];
    do
      *v14++ = v8;
    while ( (__int64 *)v7 != v14 );
  }
  else if ( v7 )
  {
    v7 = (unsigned __int64)&v18[v7];
    v13 = v18;
    do
      *v13++ = v8;
    while ( (__int64 *)v7 != v13 );
  }
  LODWORD(v17) = v9;
  sub_2B38BA0((__int64)&v16, (__int64)a1, v7, v4, v5, v6);
  v10 = (unsigned int)v17;
  v11 = 0;
  if ( (_DWORD)v17 )
  {
    do
    {
      v12 = *(int *)(a2 + 4 * v11);
      if ( (_DWORD)v12 != -1 )
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v12) = v16[v11];
      ++v11;
    }
    while ( v10 != v11 );
  }
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
}
