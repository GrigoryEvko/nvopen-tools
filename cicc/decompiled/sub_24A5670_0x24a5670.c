// Function: sub_24A5670
// Address: 0x24a5670
//
__int64 __fastcall sub_24A5670(__int64 a1, int a2)
{
  _QWORD *v2; // r14
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  _QWORD *v6; // rax
  __int64 v7; // r13
  char v8; // al
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-88h]
  unsigned __int64 v19; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-78h]
  __int64 v21[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v22[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h]
  __int64 v24; // [rsp+48h] [rbp-48h]
  __int64 v25; // [rsp+50h] [rbp-40h]

  v2 = (_QWORD *)sub_BCB2E0(*(_QWORD **)a1);
  v3 = 0x30000000000000ALL;
  if ( a2 != 2 )
    v3 = 0x10000000000000ALL;
  if ( byte_4FEB788 )
    v3 |= 0x400000000000000uLL;
  if ( (_BYTE)qword_4FEB6A8 )
    v3 |= 0x80000000000000uLL;
  if ( unk_4FE76C8 || unk_4FE7468 == 1 )
    v3 |= 0x800000000000000uLL;
  if ( (_BYTE)qword_4FEB5C8 )
    v3 |= 0x3000000000000000uLL;
  if ( (_BYTE)qword_4FEB4E8 )
    v3 |= 0x1000000000000000uLL;
  if ( (_BYTE)qword_4FEB328 )
    v3 |= 0x8000000000000000LL;
  v20 = 64;
  v19 = v3;
  v4 = sub_AD6220((__int64)v2, (__int64)&v19);
  v21[1] = 26;
  v5 = v4;
  BYTE4(v18) = 0;
  LOWORD(v23) = 261;
  v21[0] = (__int64)"__llvm_profile_raw_version";
  v6 = sub_BD2C40(88, unk_3F0FAE8);
  v7 = (__int64)v6;
  if ( v6 )
    sub_B30000((__int64)v6, a1, v2, 1, 4, v5, (__int64)v21, 0, 0, v18, 0);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  v8 = *(_BYTE *)(v7 + 32) & 0xCF | 0x10;
  *(_BYTE *)(v7 + 32) = v8;
  if ( (v8 & 0xF) != 9 )
    *(_BYTE *)(v7 + 33) |= 0x40u;
  v9 = *(_BYTE **)(a1 + 232);
  v10 = *(_QWORD *)(a1 + 240);
  v21[0] = (__int64)v22;
  sub_24A3020(v21, v9, (__int64)&v9[v10]);
  v11 = *(unsigned int *)(a1 + 284);
  v23 = *(_QWORD *)(a1 + 264);
  v24 = *(_QWORD *)(a1 + 272);
  v25 = *(_QWORD *)(a1 + 280);
  if ( (unsigned int)v11 > 8 || (v17 = 292, !_bittest64(&v17, v11)) )
  {
    v12 = *(_BYTE *)(v7 + 32);
    *(_BYTE *)(v7 + 32) = v12 & 0xF0;
    if ( (v12 & 0x30) != 0 )
      *(_BYTE *)(v7 + 33) |= 0x40u;
    v13 = sub_BAA410(a1, "__llvm_profile_raw_version", 0x1Au);
    sub_B2F990(v7, v13, v14, v15);
  }
  if ( (_QWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0]);
  return v7;
}
