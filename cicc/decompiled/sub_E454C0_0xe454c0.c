// Function: sub_E454C0
// Address: 0xe454c0
//
_QWORD *__fastcall sub_E454C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  unsigned __int64 v8; // r13
  __int64 **v9; // rbx
  unsigned __int64 v10; // r12
  _BYTE *v11; // r15
  __int64 *v12; // rdi
  _QWORD *v13; // rsi
  int v14; // edx
  _QWORD *v15; // rdi
  _BYTE *v17; // [rsp+10h] [rbp-70h] BYREF
  __int64 v18; // [rsp+18h] [rbp-68h]
  _BYTE v19[96]; // [rsp+20h] [rbp-60h] BYREF

  if ( !BYTE4(a2) )
  {
    v7 = (_QWORD *)a1;
    if ( (_DWORD)a2 == 1 )
      return v7;
  }
  v8 = *(unsigned int *)(a1 + 12);
  v9 = *(__int64 ***)(a1 + 16);
  v17 = v19;
  v10 = v8;
  v18 = 0x600000000LL;
  if ( v8 > 6 )
  {
    sub_C8D5F0((__int64)&v17, v19, v8, 8u, a5, a6);
    v11 = &v17[8 * (unsigned int)v18];
    goto LABEL_6;
  }
  if ( 8 * v8 )
  {
    v11 = v19;
    do
    {
LABEL_6:
      v12 = *v9;
      v11 += 8;
      ++v9;
      *((_QWORD *)v11 - 1) = sub_BCE1B0(v12, a2);
      --v10;
    }
    while ( v10 );
    v13 = v17;
    v14 = v18;
    goto LABEL_8;
  }
  v13 = v19;
  v14 = 0;
LABEL_8:
  v15 = *(_QWORD **)a1;
  LODWORD(v18) = v14 + v8;
  v7 = sub_BD0B90(v15, v13, (unsigned int)(v14 + v8), 0);
  if ( v17 != v19 )
    _libc_free(v17, v13);
  return v7;
}
