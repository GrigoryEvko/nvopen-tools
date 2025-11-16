// Function: sub_A79F10
// Address: 0xa79f10
//
unsigned __int64 __fastcall sub_A79F10(__int64 *a1, unsigned int a2, int *a3, __int64 a4)
{
  unsigned __int64 v4; // r15
  int *v5; // rbx
  int *v6; // r13
  __int64 v7; // r10
  __int64 v8; // rax
  int v9; // edx
  unsigned int *v10; // rax
  unsigned int *v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v13; // r12
  unsigned __int64 v15; // r11
  unsigned __int64 *v16; // rax
  unsigned __int64 v17; // [rsp+8h] [rbp-D8h]
  __int64 v18; // [rsp+10h] [rbp-D0h]
  unsigned int *v19; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-B8h]
  _BYTE v21[176]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = &a3[a4];
  v19 = (unsigned int *)v21;
  v20 = 0x800000000LL;
  if ( v5 == a3 )
  {
    v11 = (unsigned int *)v21;
    v12 = 0;
  }
  else
  {
    v6 = a3;
    do
    {
      v7 = sub_A778C0(a1, *v6, 0);
      v8 = (unsigned int)v20;
      v9 = v20;
      if ( (unsigned int)v20 >= (unsigned __int64)HIDWORD(v20) )
      {
        v15 = v4 & 0xFFFFFFFF00000000LL | a2;
        v4 = v15;
        if ( HIDWORD(v20) < (unsigned __int64)(unsigned int)v20 + 1 )
        {
          v17 = v15;
          v18 = v7;
          sub_C8D5F0(&v19, v21, (unsigned int)v20 + 1LL, 16);
          v8 = (unsigned int)v20;
          v15 = v17;
          v7 = v18;
        }
        v16 = (unsigned __int64 *)&v19[4 * v8];
        *v16 = v15;
        v16[1] = v7;
        LODWORD(v20) = v20 + 1;
      }
      else
      {
        v10 = &v19[4 * (unsigned int)v20];
        if ( v10 )
        {
          *v10 = a2;
          *((_QWORD *)v10 + 1) = v7;
          v9 = v20;
        }
        LODWORD(v20) = v9 + 1;
      }
      ++v6;
    }
    while ( v5 != v6 );
    v11 = v19;
    v12 = (unsigned int)v20;
  }
  v13 = sub_A79CA0(a1, v11, v12);
  if ( v19 != (unsigned int *)v21 )
    _libc_free(v19, v11);
  return v13;
}
