// Function: sub_13CCF60
// Address: 0x13ccf60
//
__int64 __fastcall sub_13CCF60(int a1, __int64 **a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 **v7; // r13
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 **v13; // r9
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdi
  int v22; // edx
  int v23; // r10d
  char *v24; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+28h] [rbp-58h]
  _BYTE v27[2]; // [rsp+3Eh] [rbp-42h] BYREF
  _QWORD v28[2]; // [rsp+40h] [rbp-40h] BYREF
  char v29; // [rsp+50h] [rbp-30h] BYREF

  v4 = a4[3];
  if ( !v4 )
    return v4;
  v5 = a4[4];
  if ( !v5 || !*(_QWORD *)(v5 + 40) )
    return 0;
  v28[0] = a2;
  v7 = a2;
  v8 = a3;
  v24 = (char *)v28;
  v28[1] = a3;
  while ( 1 )
  {
    if ( !*(_BYTE *)(v4 + 184) )
      sub_14CDF70(v4);
    v9 = *(unsigned int *)(v4 + 176);
    if ( !(_DWORD)v9 )
      goto LABEL_22;
    v10 = *(_QWORD *)(v4 + 160);
    v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = v10 + 88LL * v11;
    v13 = *(__int64 ***)(v12 + 24);
    if ( a2 == v13 )
      break;
    v22 = 1;
    while ( v13 != (__int64 **)-8LL )
    {
      v23 = v22 + 1;
      v11 = (v9 - 1) & (v11 + v22);
      v12 = v10 + 88LL * v11;
      v13 = *(__int64 ***)(v12 + 24);
      if ( a2 == v13 )
        goto LABEL_9;
      v22 = v23;
    }
LABEL_22:
    v24 += 8;
    if ( v24 == &v29 )
      return 0;
    v4 = a4[3];
    a2 = *(__int64 ***)v24;
  }
LABEL_9:
  if ( v12 == v10 + 88 * v9 )
    goto LABEL_22;
  v14 = *(_QWORD *)(v12 + 40);
  v26 = v14 + 32LL * *(unsigned int *)(v12 + 48);
  if ( v26 == v14 )
    goto LABEL_22;
  while ( 1 )
  {
    v15 = *(_QWORD *)(v14 + 16);
    if ( v15 )
    {
      sub_14BC8D0(
        (unsigned int)v27,
        *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)),
        a1,
        (_DWORD)v7,
        v8,
        *a4,
        1,
        0);
      if ( v27[1] )
      {
        if ( (unsigned __int8)sub_14AFF20(v15, a4[4], a4[2]) )
          break;
      }
    }
    v14 += 32;
    if ( v26 == v14 )
      goto LABEL_22;
  }
  v16 = v27[0];
  v17 = **v7;
  if ( *((_BYTE *)*v7 + 8) == 16 )
  {
    v18 = (*v7)[4];
    v19 = sub_1643320(v17);
    v20 = sub_16463B0(v19, (unsigned int)v18);
  }
  else
  {
    v20 = sub_1643320(v17);
  }
  return sub_15A0680(v20, v16, 0);
}
