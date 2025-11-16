// Function: sub_24522D0
// Address: 0x24522d0
//
void __fastcall sub_24522D0(__int64 a1)
{
  unsigned __int64 v2; // rax
  _QWORD *v3; // rdi
  int v4; // edx
  _QWORD *v5; // r14
  __int64 v6; // r15
  _QWORD *v7; // rax
  __int64 v8; // rbx
  char v9; // dl
  char v10; // al
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  _DWORD v19[6]; // [rsp+8h] [rbp-88h] BYREF
  const char *v20; // [rsp+20h] [rbp-70h] BYREF
  __int64 v21; // [rsp+28h] [rbp-68h]
  _QWORD v22[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h]
  __int64 v24; // [rsp+48h] [rbp-48h]
  __int64 v25; // [rsp+50h] [rbp-40h]

  v2 = sub_2450190();
  v3 = *(_QWORD **)a1;
  *(_QWORD *)&v19[3] = v2;
  v19[5] = v4;
  if ( (_BYTE)v4 )
  {
    LODWORD(v21) = 16;
    v5 = (_QWORD *)sub_BCB2C0(v3);
  }
  else
  {
    LODWORD(v21) = 32;
    v5 = (_QWORD *)sub_BCB2D0(v3);
  }
  v20 = 0;
  v6 = sub_AD6220((__int64)v5, (__int64)&v20);
  if ( (unsigned int)v21 > 0x40 && v20 )
    j_j___libc_free_0_0((unsigned __int64)v20);
  v21 = 23;
  LOWORD(v23) = 261;
  v20 = "__llvm_profile_sampling";
  LOBYTE(v19[1]) = 0;
  v7 = sub_BD2C40(88, unk_3F0FAE8);
  v8 = (__int64)v7;
  if ( v7 )
    sub_B30000((__int64)v7, a1, v5, 0, 4, v6, (__int64)&v20, 0, 0, *(__int64 *)v19, 0);
  v9 = *(_BYTE *)(v8 + 32);
  *(_BYTE *)(v8 + 32) = v9 & 0xCF;
  v10 = *(_BYTE *)(v8 + 33);
  if ( (v9 & 0xFu) - 7 <= 1 )
  {
    v10 |= 0x40u;
    *(_BYTE *)(v8 + 33) = v10;
  }
  *(_BYTE *)(v8 + 33) = v10 & 0xE3 | 4;
  v11 = *(_BYTE **)(a1 + 232);
  v12 = *(_QWORD *)(a1 + 240);
  v20 = (const char *)v22;
  sub_2450530((__int64 *)&v20, v11, (__int64)&v11[v12]);
  v13 = *(unsigned int *)(a1 + 284);
  v23 = *(_QWORD *)(a1 + 264);
  v24 = *(_QWORD *)(a1 + 272);
  v25 = *(_QWORD *)(a1 + 280);
  if ( (unsigned int)v13 > 8 || (v18 = 292, !_bittest64(&v18, v13)) )
  {
    v14 = *(_BYTE *)(v8 + 32);
    *(_BYTE *)(v8 + 32) = v14 & 0xF0;
    if ( (v14 & 0x30) != 0 )
      *(_BYTE *)(v8 + 33) |= 0x40u;
    v15 = sub_BAA410(a1, "__llvm_profile_sampling", 0x17u);
    sub_B2F990(v8, v15, v16, v17);
  }
  *(_QWORD *)v19 = v8;
  sub_2A41DC0(a1, v19, 1);
  if ( v20 != (const char *)v22 )
    j_j___libc_free_0((unsigned __int64)v20);
}
