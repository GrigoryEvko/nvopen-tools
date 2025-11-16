// Function: sub_1560040
// Address: 0x1560040
//
__int64 __fastcall sub_1560040(__int64 *a1, int a2, unsigned int *a3, __int64 a4)
{
  unsigned int *v4; // rbx
  unsigned int *v5; // r15
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // r9
  int *v9; // rax
  __int64 v10; // rdx
  int *v11; // rsi
  __int64 v12; // r12
  __int64 v14; // [rsp+0h] [rbp-D0h]
  int *v15; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+18h] [rbp-B8h]
  _BYTE v17[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = &a3[a4];
  v15 = (int *)v17;
  v16 = 0x800000000LL;
  if ( v4 == a3 )
  {
    v10 = 0;
    v11 = (int *)v17;
  }
  else
  {
    v5 = a3;
    do
    {
      v6 = sub_155CEC0(a1, *v5, 0);
      v7 = v16;
      v8 = v6;
      if ( (unsigned int)v16 >= HIDWORD(v16) )
      {
        v14 = v6;
        sub_16CD150(&v15, v17, 0, 16);
        v7 = v16;
        v8 = v14;
      }
      v9 = &v15[4 * v7];
      if ( v9 )
      {
        *v9 = a2;
        *((_QWORD *)v9 + 1) = v8;
        v7 = v16;
      }
      v10 = (unsigned int)(v7 + 1);
      ++v5;
      LODWORD(v16) = v10;
    }
    while ( v4 != v5 );
    v11 = v15;
  }
  v12 = sub_155FBC0(a1, v11, v10);
  if ( v15 != (int *)v17 )
    _libc_free((unsigned __int64)v15);
  return v12;
}
