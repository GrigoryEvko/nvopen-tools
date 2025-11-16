// Function: sub_FDE3B0
// Address: 0xfde3b0
//
__int64 __fastcall sub_FDE3B0(_QWORD *a1, _QWORD *a2, unsigned int *a3)
{
  unsigned __int64 v3; // r10
  _QWORD *v4; // r14
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 result; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rbx
  int v12; // eax
  unsigned __int64 v13; // r10
  unsigned int v14; // r13d
  __int64 v15; // rax
  int v16; // [rsp+14h] [rbp-CCh]
  __int64 v17; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v19; // [rsp+30h] [rbp-B0h]
  unsigned __int8 v20; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v21; // [rsp+38h] [rbp-A8h]
  __int64 v22; // [rsp+38h] [rbp-A8h]
  int v23; // [rsp+4Ch] [rbp-94h] BYREF
  _QWORD v24[2]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v25[64]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v26; // [rsp+A0h] [rbp-40h]
  char v27; // [rsp+A8h] [rbp-38h]

  v4 = a2;
  v5 = a1[8];
  v27 = 0;
  v6 = *a3;
  v24[0] = v25;
  v24[1] = 0x400000000LL;
  v26 = 0;
  v7 = *(__int64 **)(v5 + 24 * v6 + 8);
  if ( v7 && *((_BYTE *)v7 + 8) )
  {
    do
    {
      v8 = v7;
      v7 = (__int64 *)*v7;
    }
    while ( v7 && *((_BYTE *)v7 + 8) );
    result = sub_FE8E10(a1, a2, v8, v24);
    if ( !(_BYTE)result )
      goto LABEL_8;
LABEL_7:
    a2 = a3;
    sub_FEA740(a1, a3, v4, v24);
    result = 1;
    goto LABEL_8;
  }
  v17 = *(_QWORD *)(a1[17] + 8 * v6);
  v10 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == v17 + 48 )
    goto LABEL_7;
  if ( !v10 )
    BUG();
  v11 = v10 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
    goto LABEL_7;
  v21 = v3;
  v12 = sub_B46E30(v11);
  v13 = v21;
  v16 = v12;
  if ( !v12 )
    goto LABEL_7;
  v14 = 0;
  while ( 1 )
  {
    v19 = v14 | v13 & 0xFFFFFFFF00000000LL;
    v22 = (unsigned int)sub_FF0420(a1[14], v17, v11, v19);
    v15 = sub_B46EC0(v11, v14);
    a2 = v24;
    v23 = sub_FDD0F0((__int64)a1, v15);
    result = sub_FE8BD0(a1, v24, v4, a3, &v23, v22);
    if ( !(_BYTE)result )
      break;
    ++v14;
    v13 = v19;
    if ( v16 == v14 )
      goto LABEL_7;
  }
LABEL_8:
  if ( (_BYTE *)v24[0] != v25 )
  {
    v20 = result;
    _libc_free(v24[0], a2);
    return v20;
  }
  return result;
}
