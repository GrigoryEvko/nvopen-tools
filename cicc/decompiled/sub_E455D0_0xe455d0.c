// Function: sub_E455D0
// Address: 0xe455d0
//
_QWORD *__fastcall sub_E455D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rbx
  _BYTE *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rsi
  int v13; // eax
  __int64 v14; // rdx
  _QWORD *v15; // rdi
  _QWORD *v16; // r12
  _BYTE *v18; // [rsp+0h] [rbp-70h] BYREF
  __int64 v19; // [rsp+8h] [rbp-68h]
  _BYTE v20[96]; // [rsp+10h] [rbp-60h] BYREF

  v6 = *(unsigned int *)(a1 + 12);
  v7 = *(_QWORD *)(a1 + 16);
  v18 = v20;
  v19 = 0x600000000LL;
  v8 = 8 * v6;
  if ( v6 > 6 )
  {
    sub_C8D5F0((__int64)&v18, v20, v6, 8u, a5, a6);
    v9 = &v18[8 * (unsigned int)v19];
  }
  else
  {
    if ( !v8 )
    {
      v13 = v6;
      v14 = v6;
      v12 = v20;
      goto LABEL_9;
    }
    v9 = v20;
  }
  v10 = 0;
  do
  {
    v11 = *(_QWORD *)(v7 + v10);
    if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
      v11 = **(_QWORD **)(v11 + 16);
    *(_QWORD *)&v9[v10] = v11;
    v10 += 8;
  }
  while ( v8 != v10 );
  v12 = v18;
  v13 = v6 + v19;
  v14 = (unsigned int)(v6 + v19);
LABEL_9:
  v15 = *(_QWORD **)a1;
  LODWORD(v19) = v13;
  v16 = sub_BD0B90(v15, v12, v14, 0);
  if ( v18 != v20 )
    _libc_free(v18, v12);
  return v16;
}
