// Function: sub_29BAF70
// Address: 0x29baf70
//
double __fastcall sub_29BAF70(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned __int64 v7; // r12
  __int64 *v8; // rdi
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx
  __int64 v12; // rax
  double result; // xmm0_8
  __int64 v15; // [rsp+8h] [rbp-68h]
  _QWORD *v16; // [rsp+10h] [rbp-60h] BYREF
  __int64 v17; // [rsp+18h] [rbp-58h]
  _QWORD v18[10]; // [rsp+20h] [rbp-50h] BYREF

  v4 = a3;
  v7 = a2;
  v8 = v18;
  v16 = v18;
  v17 = 0x600000000LL;
  if ( !a2 )
    goto LABEL_12;
  v9 = v18;
  v10 = v18;
  if ( a2 > 6 )
  {
    v15 = a4;
    sub_C8D5F0((__int64)&v16, v18, a2, 8u, a3, a4);
    v10 = v16;
    v4 = a3;
    a4 = v15;
    v9 = &v16[(unsigned int)v17];
    v11 = &v16[a2];
    if ( v11 != v9 )
      goto LABEL_4;
  }
  else
  {
    v11 = &v18[a2];
    if ( v11 != v18 )
    {
      do
      {
LABEL_4:
        if ( v9 )
          *v9 = 0;
        ++v9;
      }
      while ( v11 != v9 );
      v10 = v16;
    }
  }
  LODWORD(v17) = a2;
  v12 = 0;
  while ( 1 )
  {
    v10[v12] = v12;
    if ( a2 == ++v12 )
      break;
    v10 = v16;
  }
  v8 = v16;
  a2 = (unsigned int)v17;
LABEL_12:
  result = sub_29BAC40(v8, a2, a1, v7, v4, a4);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
  return result;
}
