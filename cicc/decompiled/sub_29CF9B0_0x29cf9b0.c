// Function: sub_29CF9B0
// Address: 0x29cf9b0
//
__int64 __fastcall sub_29CF9B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r8
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned __int64 v14; // r12
  __int64 *v15; // rsi
  __int64 **v16; // rdi
  char v17; // al
  __int64 v18; // r12
  __int64 *v20; // [rsp+10h] [rbp-140h] BYREF
  __int64 v21; // [rsp+18h] [rbp-138h]
  _BYTE v22[304]; // [rsp+20h] [rbp-130h] BYREF

  v21 = 0x2000000000LL;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(unsigned int *)(a1 + 16);
  v20 = (__int64 *)v22;
  v9 = (_QWORD *)(v7 + 8 * v8);
  if ( v9 == (_QWORD *)v7 )
  {
    v13 = 0;
    v15 = (__int64 *)v22;
  }
  else
  {
    v10 = (_QWORD *)v7;
    do
    {
      v14 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !*v10 || (*v10 & 4) != 0 || !v14 )
        v14 = sub_29CF9B0(*v10 & 0xFFFFFFFFFFFFFFF8LL);
      v11 = (unsigned int)v21;
      v12 = (unsigned int)v21 + 1LL;
      if ( v12 > HIDWORD(v21) )
      {
        sub_C8D5F0((__int64)&v20, v22, v12, 8u, v7, a6);
        v11 = (unsigned int)v21;
      }
      ++v10;
      v20[v11] = v14;
      v13 = (unsigned int)(v21 + 1);
      LODWORD(v21) = v21 + 1;
    }
    while ( v9 != v10 );
    v15 = v20;
  }
  v16 = *(__int64 ***)a1;
  v17 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v17 == 15 )
  {
    v18 = sub_AD24A0(v16, v15, v13);
  }
  else if ( v17 == 16 )
  {
    v18 = sub_AD1300(v16, v15, v13);
  }
  else
  {
    v18 = sub_AD3730(v15, v13);
  }
  if ( v20 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v20);
  return v18;
}
