// Function: sub_29459F0
// Address: 0x29459f0
//
__int64 __fastcall sub_29459F0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // r13
  void *v12; // r14
  unsigned int v13; // r13d
  _QWORD *v14; // r14
  _QWORD *v15; // r12
  __int64 v16; // rax
  unsigned int v18; // [rsp+8h] [rbp-4E8h]
  unsigned int v19; // [rsp+Ch] [rbp-4E4h]
  __int64 v20; // [rsp+10h] [rbp-4E0h]
  __int16 v21; // [rsp+1Ah] [rbp-4D6h]
  int v22; // [rsp+1Ch] [rbp-4D4h]
  __int64 v23; // [rsp+20h] [rbp-4D0h]
  char v24[8]; // [rsp+30h] [rbp-4C0h] BYREF
  int v25; // [rsp+38h] [rbp-4B8h] BYREF
  _QWORD *v26; // [rsp+40h] [rbp-4B0h]
  int *v27; // [rsp+48h] [rbp-4A8h]
  int *v28; // [rsp+50h] [rbp-4A0h]
  __int64 v29; // [rsp+58h] [rbp-498h]
  _BYTE *v30; // [rsp+60h] [rbp-490h]
  __int64 v31; // [rsp+68h] [rbp-488h]
  _BYTE v32[264]; // [rsp+70h] [rbp-480h] BYREF
  _BYTE *v33; // [rsp+178h] [rbp-378h]
  __int64 v34; // [rsp+180h] [rbp-370h]
  _BYTE v35[768]; // [rsp+188h] [rbp-368h] BYREF
  __int64 v36; // [rsp+488h] [rbp-68h]
  __int64 v37; // [rsp+490h] [rbp-60h]
  __int16 v38; // [rsp+498h] [rbp-58h]
  int v39; // [rsp+49Ch] [rbp-54h]
  __int64 v40; // [rsp+4A0h] [rbp-50h]
  void *v41; // [rsp+4A8h] [rbp-48h]
  unsigned __int64 v42; // [rsp+4B0h] [rbp-40h]
  __int64 v43; // [rsp+4B8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_30;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v6 = *(__int64 **)(a1 + 8);
  v20 = v5 + 176;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F89C28 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_29;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F89C28);
  v23 = sub_DFED00(v9, a2);
  v22 = *(_DWORD *)(a1 + 176);
  v21 = *(_WORD *)(a1 + 180);
  sub_C7D6A0(0, 0, 4);
  v10 = *(_DWORD *)(a1 + 208);
  if ( v10 )
  {
    v11 = 4LL * v10;
    v12 = (void *)sub_C7D670(v11, 4);
    v19 = *(_DWORD *)(a1 + 200);
    v18 = *(_DWORD *)(a1 + 204);
    memcpy(v12, *(const void **)(a1 + 192), v11);
  }
  else
  {
    v18 = 0;
    v11 = 0;
    v12 = 0;
    v19 = 0;
  }
  v27 = &v25;
  v28 = &v25;
  v31 = 0x1000000000LL;
  v33 = v35;
  v34 = 0x2000000000LL;
  v25 = 0;
  v36 = v20;
  v26 = 0;
  v37 = v23;
  v29 = 0;
  v38 = v21;
  v30 = v32;
  v39 = v22;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  sub_C7D6A0(0, 0, 4);
  LODWORD(v43) = v10;
  if ( v10 )
  {
    v41 = (void *)sub_C7D670(v11, 4);
    v42 = __PAIR64__(v18, v19);
    memcpy(v41, v12, 4LL * (unsigned int)v43);
  }
  else
  {
    v41 = 0;
    v42 = 0;
  }
  sub_C7D6A0((__int64)v12, v11, 4);
  v13 = sub_2942CE0((__int64)v24, a2);
  sub_C7D6A0((__int64)v41, 4LL * (unsigned int)v43, 4);
  v14 = v33;
  v15 = &v33[24 * (unsigned int)v34];
  if ( v33 != (_BYTE *)v15 )
  {
    do
    {
      v16 = *(v15 - 1);
      v15 -= 3;
      if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        sub_BD60C0(v15);
    }
    while ( v14 != v15 );
    v15 = v33;
  }
  if ( v15 != (_QWORD *)v35 )
    _libc_free((unsigned __int64)v15);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  sub_293AA10(v26);
  return v13;
}
