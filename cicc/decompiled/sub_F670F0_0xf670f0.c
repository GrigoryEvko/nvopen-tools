// Function: sub_F670F0
// Address: 0xf670f0
//
__int64 __fastcall sub_F670F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 *v18; // rax
  __int64 v19; // [rsp+0h] [rbp-B0h]
  __int64 v20; // [rsp+8h] [rbp-A8h]
  __int64 v21; // [rsp+10h] [rbp-A0h]
  __int64 v24; // [rsp+28h] [rbp-88h]
  _QWORD v25[8]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v26; // [rsp+70h] [rbp-40h]

  v24 = sub_AA4E30(**(_QWORD **)(a1 + 32));
  v8 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 56LL);
  v19 = a1 + 56;
  while ( 1 )
  {
    if ( !v8 )
      BUG();
    v9 = v8 - 24;
    if ( *(_BYTE *)(v8 - 24) != 84 )
      return 0;
    v10 = *(_QWORD *)(v8 + 8);
    v25[1] = 0;
    v25[0] = v24;
    v25[2] = 0;
    v25[3] = a2;
    memset(&v25[5], 0, 24);
    v25[4] = a3;
    v26 = 257;
    v11 = sub_1020E10(v8 - 24, v25, v4, v5, v6, v7);
    if ( v11 )
    {
      sub_BD84D0(v8 - 24, v11);
      sub_B43D60((_QWORD *)(v8 - 24));
      goto LABEL_3;
    }
    v6 = *(_DWORD *)(v8 - 20) & 0x7FFFFFF;
    if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) != 0 )
      break;
LABEL_3:
    v8 = v10;
  }
  v4 = 0;
  v6 = 8LL * (unsigned int)v6;
  while ( 1 )
  {
    v12 = *(_QWORD *)(v8 - 32);
    v13 = *(_QWORD *)(v12 + 4 * v4);
    if ( !v13 || v9 != v13 )
      goto LABEL_9;
    v14 = *(_QWORD *)(32LL * *(unsigned int *)(v8 + 48) + v12 + v4);
    if ( *(_BYTE *)(a1 + 84) )
      break;
    v20 = v6;
    v21 = v4;
    v18 = sub_C8CA60(v19, v14);
    v4 = v21;
    v6 = v20;
    if ( v18 )
      return v9;
LABEL_9:
    v4 += 8;
    if ( v6 == v4 )
      goto LABEL_3;
  }
  v15 = *(_QWORD **)(a1 + 64);
  v16 = &v15[*(unsigned int *)(a1 + 76)];
  if ( v15 == v16 )
    goto LABEL_9;
  while ( v14 != *v15 )
  {
    if ( v16 == ++v15 )
      goto LABEL_9;
  }
  return v9;
}
