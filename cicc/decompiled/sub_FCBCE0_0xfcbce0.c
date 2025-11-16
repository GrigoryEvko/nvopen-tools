// Function: sub_FCBCE0
// Address: 0xfcbce0
//
void __fastcall sub_FCBCE0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int *v9; // rdi
  unsigned int *v10; // r12
  unsigned int *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int *v19; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-B8h]
  _BYTE v21[176]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = (__int64)&v19;
  v19 = (unsigned int *)v21;
  v20 = 0x800000000LL;
  sub_B9A9D0(a2, (__int64)&v19);
  sub_B91E30(a2, (__int64)&v19);
  v9 = v19;
  v10 = &v19[4 * (unsigned int)v20];
  if ( v10 != v19 )
  {
    v11 = v19;
    do
    {
      v13 = *((_QWORD *)v11 + 1);
      v14 = sub_FC95E0(a1, v13, v5, v6, v7, v8);
      if ( (_BYTE)v15 )
        v12 = (__int64)v14;
      else
        v12 = sub_FCBA10(a1, v13, v15, v16, v17, v18);
      v3 = *v11;
      v11 += 4;
      sub_B994D0(a2, v3, v12);
    }
    while ( v10 != v11 );
    v9 = v19;
  }
  if ( v9 != (unsigned int *)v21 )
    _libc_free(v9, v3);
}
