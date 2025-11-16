// Function: sub_13FD1C0
// Address: 0x13fd1c0
//
void __fastcall sub_13FD1C0(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r11
  _BYTE *v8; // rdi
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // [rsp+0h] [rbp-90h]
  _QWORD *v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  _QWORD v19[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h]
  _QWORD v22[10]; // [rsp+40h] [rbp-50h] BYREF

  v20 = v22;
  v2 = sub_13FD000(a1);
  v22[0] = 0;
  v21 = 0x400000001LL;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 8);
    if ( (unsigned int)v3 > 1 )
    {
      v4 = *(unsigned int *)(v2 + 8);
      v5 = 1;
      while ( 1 )
      {
        v7 = *(_QWORD *)(v2 + 8 * (v5 - v3));
        if ( (unsigned __int8)(*(_BYTE *)v7 - 4) > 0x1Eu )
          break;
        v8 = *(_BYTE **)(v7 - 8LL * *(unsigned int *)(v7 + 8));
        if ( *v8 )
          break;
        v9 = sub_161E970(v8);
        if ( v10 <= 0x10
          || *(_QWORD *)v9 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v9 + 8) ^ 0x6C6C6F726E752E70LL
          || *(_BYTE *)(v9 + 16) != 46 )
        {
          v7 = *(_QWORD *)(v2 + 8 * (v5 - *(unsigned int *)(v2 + 8)));
          v6 = (unsigned int)v21;
          if ( (unsigned int)v21 >= HIDWORD(v21) )
          {
LABEL_14:
            v16 = v7;
            sub_16CD150(&v20, v22, 0, 8);
            v6 = (unsigned int)v21;
            v7 = v16;
          }
LABEL_5:
          v20[v6] = v7;
          LODWORD(v21) = v21 + 1;
        }
        if ( v4 == ++v5 )
          goto LABEL_15;
        v3 = *(unsigned int *)(v2 + 8);
      }
      v6 = (unsigned int)v21;
      if ( (unsigned int)v21 >= HIDWORD(v21) )
        goto LABEL_14;
      goto LABEL_5;
    }
  }
LABEL_15:
  v11 = sub_157E9C0(**(_QWORD **)(a1 + 32));
  v17 = v19;
  v12 = v11;
  v18 = 0x100000001LL;
  v19[0] = sub_161FF10(v11, "llvm.loop.unroll.disable", 24);
  v13 = sub_1627350(v12, v19, 1, 0, 1);
  v14 = (unsigned int)v21;
  if ( (unsigned int)v21 >= HIDWORD(v21) )
  {
    sub_16CD150(&v20, v22, 0, 8);
    v14 = (unsigned int)v21;
  }
  v20[v14] = v13;
  LODWORD(v21) = v21 + 1;
  v15 = sub_1627350(v12, v20, (unsigned int)v21, 0, 1);
  sub_1630830(v15, 0, v15);
  sub_13FCC30(a1, v15);
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
}
