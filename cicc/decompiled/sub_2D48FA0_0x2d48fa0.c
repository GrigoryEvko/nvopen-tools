// Function: sub_2D48FA0
// Address: 0x2d48fa0
//
__int64 __fastcall sub_2D48FA0(unsigned __int64 a1, __int64 a2)
{
  __int16 v3; // bx
  __int64 v4; // r12
  __int16 v5; // r15
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  int v8; // ebx
  _QWORD *v9; // r14
  unsigned int *v10; // rbx
  unsigned int *v11; // r12
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // rax
  __int16 v16; // [rsp+Ch] [rbp-194h]
  __int64 v17; // [rsp+10h] [rbp-190h]
  int v18; // [rsp+2Ch] [rbp-174h] BYREF
  _QWORD v19[4]; // [rsp+30h] [rbp-170h] BYREF
  __int16 v20; // [rsp+50h] [rbp-150h]
  unsigned int *v21; // [rsp+60h] [rbp-140h] BYREF
  int v22; // [rsp+68h] [rbp-138h]
  char v23; // [rsp+70h] [rbp-130h] BYREF
  __int64 v24; // [rsp+98h] [rbp-108h]
  __int64 v25; // [rsp+A0h] [rbp-100h]
  __int64 v26; // [rsp+B8h] [rbp-E8h]
  void *v27; // [rsp+E0h] [rbp-C0h]
  void *v28; // [rsp+E8h] [rbp-B8h]
  _QWORD v29[12]; // [rsp+140h] [rbp-60h] BYREF

  sub_2D46B10((__int64)&v21, a2, a1);
  v3 = (*(_WORD *)(a2 + 2) >> 7) & 7;
  if ( v3 != 1 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    v5 = (*(_WORD *)(a2 + 2) >> 7) & 7;
    v6 = sub_AD6530(*(_QWORD *)(a2 + 8), a2);
    switch ( v3 )
    {
      case 0:
      case 1:
      case 3:
        BUG();
      case 2:
      case 5:
        goto LABEL_4;
      case 4:
      case 6:
        v16 = 4;
        goto LABEL_5;
      case 7:
        v16 = 7;
        v5 = 7;
        goto LABEL_5;
    }
  }
  v4 = *(_QWORD *)(a2 - 32);
  v5 = 2;
  v6 = sub_AD6530(*(_QWORD *)(a2 + 8), a2);
LABEL_4:
  v16 = 2;
LABEL_5:
  v17 = v6;
  _BitScanReverse64(&v7, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v20 = 257;
  v8 = (unsigned __int8)(63 - (v7 ^ 0x3F));
  v9 = sub_BD2C40(80, unk_3F148C4);
  if ( v9 )
    sub_B4D5A0((__int64)v9, v4, v17, v17, v8, v5, v16, 1, 0, 0);
  (*(void (__fastcall **)(__int64, _QWORD *, _QWORD *, __int64, __int64))(*(_QWORD *)v26 + 16LL))(
    v26,
    v9,
    v19,
    v24,
    v25);
  v10 = v21;
  v11 = &v21[4 * v22];
  if ( v21 != v11 )
  {
    do
    {
      v12 = *((_QWORD *)v10 + 1);
      v13 = *v10;
      v10 += 4;
      sub_B99FD0((__int64)v9, v13, v12);
    }
    while ( v11 != v10 );
  }
  v19[0] = "loaded";
  v20 = 259;
  v18 = 0;
  v14 = sub_94D3D0(&v21, (__int64)v9, (__int64)&v18, 1, (__int64)v19);
  sub_BD84D0(a2, v14);
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v29);
  v27 = &unk_49E5698;
  v28 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v21 != (unsigned int *)&v23 )
    _libc_free((unsigned __int64)v21);
  return 1;
}
