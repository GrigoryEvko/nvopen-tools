// Function: sub_AE6F30
// Address: 0xae6f30
//
__int64 __fastcall sub_AE6F30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // r13d
  __int64 v7; // rax
  unsigned int v8; // ebx
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  __int64 v12; // rbx
  bool v13; // al
  __int64 v14; // r15
  __int64 v15; // rdx
  unsigned __int64 *v16; // [rsp+8h] [rbp-68h]
  _QWORD v17[2]; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-48h]
  _QWORD v20[8]; // [rsp+30h] [rbp-40h] BYREF

  v17[0] = a4;
  v17[1] = a5;
  if ( (_BYTE)a5 )
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  v6 = a3;
  v19 = sub_AE43F0(a2, *(_QWORD *)(a3 + 8));
  if ( v19 > 0x40 )
    sub_C43690(&v18, 0, 0);
  else
    v18 = 0;
  v7 = sub_BD45C0(v6, a2, (unsigned int)&v18, 1, 0, 0, 0, 0);
  v8 = v19;
  v9 = v7;
  v10 = 1LL << ((unsigned __int8)v19 - 1);
  if ( v19 > 0x40 )
  {
    if ( (*(_QWORD *)(v18 + 8LL * ((v19 - 1) >> 6)) & v10) != 0 )
      goto LABEL_10;
    v16 = (unsigned __int64 *)v18;
    if ( v8 - (unsigned int)sub_C444A0(&v18) > 0x40 )
      goto LABEL_10;
    v11 = *v16;
  }
  else
  {
    v11 = v18;
    if ( (v18 & v10) != 0 )
      goto LABEL_10;
  }
  if ( v11 != -1 && *(_BYTE *)v9 == 60 )
  {
    v12 = sub_CA1930(v17);
    v13 = 0;
    v14 = 8 * v11;
    if ( !v14 )
    {
      v20[0] = sub_9208B0(a2, *(_QWORD *)(v9 + 72));
      v20[1] = v15;
      v13 = v12 == sub_CA1930(v20);
    }
    *(_QWORD *)(a1 + 16) = v12;
    v8 = v19;
    *(_QWORD *)a1 = v9;
    *(_QWORD *)(a1 + 8) = v14;
    *(_BYTE *)(a1 + 24) = v13;
    *(_BYTE *)(a1 + 32) = 1;
    goto LABEL_11;
  }
LABEL_10:
  *(_BYTE *)(a1 + 32) = 0;
LABEL_11:
  if ( v8 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
