// Function: sub_26DC3F0
// Address: 0x26dc3f0
//
__int64 __fastcall sub_26DC3F0(_QWORD *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int128 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned int v15; // edx
  __int64 *v16; // rsi
  __int64 v17; // r8
  unsigned int v18; // ebx
  unsigned int v19; // r14d
  int v21; // esi
  int v22; // r10d
  __int64 v23; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+28h] [rbp-38h]

  if ( !unk_4F838D4 )
  {
    if ( (unsigned int)sub_26BDAB0(a2) )
      goto LABEL_8;
    return 0;
  }
  v5 = a1[149];
  v6 = a1[150];
  v25 = sub_B2D7E0(a2, "sample-profile-suffix-elision-policy", 0x24u);
  v23 = sub_A72240(&v25);
  v24 = v7;
  *(_QWORD *)&v8 = sub_BD5D20(a2);
  v9 = sub_C16140(v8, v23, v24);
  v11 = sub_B2F650(v9, v10);
  v12 = *(_QWORD *)(v5 + 8);
  v13 = v11;
  v14 = *(unsigned int *)(v5 + 24);
  if ( !(_DWORD)v14 )
    goto LABEL_17;
  v15 = (v14 - 1) & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * v13));
  v16 = (__int64 *)(v12 + 24LL * v15);
  v17 = *v16;
  if ( v13 != *v16 )
  {
    v21 = 1;
    while ( v17 != -1 )
    {
      v22 = v21 + 1;
      v15 = (v14 - 1) & (v21 + v15);
      v16 = (__int64 *)(v12 + 24LL * v15);
      v17 = *v16;
      if ( v13 == *v16 )
        goto LABEL_4;
      v21 = v22;
    }
LABEL_17:
    if ( !(unsigned __int8)sub_B2D620(a2, "profile-checksum-mismatch", 0x19u) )
      goto LABEL_8;
LABEL_7:
    if ( LOBYTE(qword_4FF8200[17]) )
      goto LABEL_8;
    return 0;
  }
LABEL_4:
  if ( v16 == (__int64 *)(v12 + 24 * v14) || (*(_BYTE *)(a2 + 32) & 0xF) == 1 )
    goto LABEL_17;
  if ( v16[2] != *(_QWORD *)(v6 + 8) )
    goto LABEL_7;
LABEL_8:
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  if ( (_BYTE)qword_4FF6E68 )
    v18 = sub_26D56A0((__int64)a1, a2, (__int64)&v25, a3);
  else
    v18 = sub_26D6270((__int64)a1, a2, (__int64)&v25);
  LOBYTE(v18) = sub_26DC0A0((__int64)a1, a2, (__int64)&v25) | v18;
  v19 = v18;
  if ( (_BYTE)v18 )
    sub_26CCA10((__int64)a1, a2);
  sub_26C2500(a1, a2);
  sub_C7D6A0(v26, 8LL * (unsigned int)v28, 8);
  return v19;
}
