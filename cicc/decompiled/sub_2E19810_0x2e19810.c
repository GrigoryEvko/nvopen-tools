// Function: sub_2E19810
// Address: 0x2e19810
//
void __fastcall sub_2E19810(_QWORD *a1, unsigned __int64 a2, char a3)
{
  unsigned __int64 v3; // rcx
  unsigned __int64 v6; // rax
  int v8; // edx
  __int64 v9; // rdi
  __int64 i; // rsi
  __int16 v11; // dx
  __int64 v12; // rsi
  __int64 v13; // r8
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r10
  __int64 v17; // r14
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // edx
  int v23; // r11d
  _QWORD v24[6]; // [rsp+0h] [rbp-B0h] BYREF
  _BYTE *v25; // [rsp+30h] [rbp-80h]
  __int64 v26; // [rsp+38h] [rbp-78h]
  int v27; // [rsp+40h] [rbp-70h]
  char v28; // [rsp+44h] [rbp-6Ch]
  _BYTE v29[104]; // [rsp+48h] [rbp-68h] BYREF

  v3 = a2;
  v6 = a2;
  v8 = *(_DWORD *)(a2 + 44);
  v9 = a1[4];
  if ( (v8 & 4) != 0 )
  {
    do
      v6 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v6 + 44) & 4) != 0 );
  }
  if ( (v8 & 8) != 0 )
  {
    do
      v3 = *(_QWORD *)(v3 + 8);
    while ( (*(_BYTE *)(v3 + 44) & 8) != 0 );
  }
  for ( i = *(_QWORD *)(v3 + 8); i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v11 = *(_WORD *)(v6 + 68);
    if ( (unsigned __int16)(v11 - 14) > 4u && v11 != 24 )
      break;
  }
  v12 = *(unsigned int *)(v9 + 144);
  v13 = *(_QWORD *)(v9 + 128);
  if ( (_DWORD)v12 )
  {
    v14 = (v12 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v15 = (__int64 *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( *v15 == v6 )
      goto LABEL_11;
    v22 = 1;
    while ( v16 != -4096 )
    {
      v23 = v22 + 1;
      v14 = (v12 - 1) & (v22 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( *v15 == v6 )
        goto LABEL_11;
      v22 = v23;
    }
  }
  v15 = (__int64 *)(v13 + 16 * v12);
LABEL_11:
  v17 = v15[1];
  sub_2FAD510(v9, a2, 0);
  v18 = sub_2E192D0(a1[4], a2, 0);
  v19 = a1[2];
  v20 = a1[1];
  v24[4] = v18;
  v24[0] = a1;
  v24[1] = v20;
  v24[2] = v19;
  v24[3] = v17;
  v24[5] = 0;
  v25 = v29;
  v26 = 8;
  v27 = 0;
  v28 = 1;
  v29[64] = a3;
  sub_2E15890((__int64)v24, a2, v19, v20, v21);
  if ( !v28 )
    _libc_free((unsigned __int64)v25);
}
