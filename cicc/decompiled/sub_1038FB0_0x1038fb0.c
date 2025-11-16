// Function: sub_1038FB0
// Address: 0x1038fb0
//
__int64 __fastcall sub_1038FB0(__int64 *a1, unsigned __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  __int64 *v18; // rsi
  __int64 v19; // r12
  __int64 *v21; // [rsp+10h] [rbp-80h] BYREF
  __int64 v22; // [rsp+18h] [rbp-78h]
  _BYTE v23[112]; // [rsp+20h] [rbp-70h] BYREF

  v7 = a1;
  v21 = (__int64 *)v23;
  v22 = 0x800000000LL;
  if ( a2 <= 8 )
  {
    v8 = &a1[a2];
    if ( v8 != a1 )
      goto LABEL_3;
LABEL_10:
    v17 = (unsigned int)v22;
    goto LABEL_6;
  }
  v8 = &a1[a2];
  sub_C8D5F0((__int64)&v21, v23, a2, 8u, a5, a6);
  if ( v8 == a1 )
    goto LABEL_10;
  do
  {
LABEL_3:
    v9 = *v7;
    v10 = sub_BCB2E0(a3);
    v11 = sub_ACD640(v10, v9, 0);
    v14 = sub_B98A20(v11, v9);
    v15 = (unsigned int)v22;
    v16 = (unsigned int)v22 + 1LL;
    if ( v16 > HIDWORD(v22) )
    {
      sub_C8D5F0((__int64)&v21, v23, v16, 8u, v12, v13);
      v15 = (unsigned int)v22;
    }
    ++v7;
    v21[v15] = (__int64)v14;
    v17 = (unsigned int)(v22 + 1);
    LODWORD(v22) = v22 + 1;
  }
  while ( v8 != v7 );
LABEL_6:
  v18 = v21;
  v19 = sub_B9C770(a3, v21, (__int64 *)v17, 0, 1);
  if ( v21 != (__int64 *)v23 )
    _libc_free(v21, v18);
  return v19;
}
