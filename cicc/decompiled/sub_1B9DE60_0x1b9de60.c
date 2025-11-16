// Function: sub_1B9DE60
// Address: 0x1b9de60
//
void __fastcall sub_1B9DE60(__int64 a1, unsigned __int64 a2, unsigned int *a3, char a4)
{
  char v7; // r14
  __int64 v8; // r14
  __int64 v9; // r13
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // r8d
  unsigned __int64 *v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // [rsp+0h] [rbp-90h]
  __int64 *v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v29[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v31; // [rsp+50h] [rbp-40h]

  v27 = (__int64 *)(a1 + 96);
  sub_1B91520(a1, (__int64 *)(a1 + 96), a2);
  v7 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  v28 = sub_15F4880(a2);
  if ( v7 )
  {
    v29[0] = sub_1649960(a2);
    v29[1] = v23;
    v30[0] = (__int64)v29;
    v31 = 773;
    v30[1] = (__int64)".cloned";
    sub_164B780(v28, v30);
  }
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v8 = 0;
    v9 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v10 = *(_QWORD *)(a2 - 8);
      else
        v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v11 = sub_1B9DCB0(a1, *(__int64 **)(v10 + v8), a3);
      if ( (*(_BYTE *)(v28 + 23) & 0x40) != 0 )
        v12 = *(_QWORD *)(v28 - 8);
      else
        v12 = v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF);
      v13 = (__int64 *)(v8 + v12);
      if ( *v13 )
      {
        v14 = v13[1];
        v15 = v13[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v15 = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
      }
      *v13 = v11;
      if ( v11 )
      {
        v16 = *(_QWORD *)(v11 + 8);
        v13[1] = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = (unsigned __int64)(v13 + 1) | *(_QWORD *)(v16 + 16) & 3LL;
        v13[2] = (v11 + 8) | v13[2] & 3;
        *(_QWORD *)(v11 + 8) = v13;
      }
      v8 += 24;
    }
    while ( v9 != v8 );
  }
  sub_1B91640(a1, v28, a2);
  v17 = *(_QWORD *)(a1 + 104);
  v18 = v28;
  v31 = 257;
  if ( v17 )
  {
    v25 = *(__int64 **)(a1 + 112);
    sub_157E9D0(v17 + 40, v28);
    v19 = *v25;
    v20 = *(_QWORD *)(v28 + 24) & 7LL;
    *(_QWORD *)(v28 + 32) = v25;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v18 + 24) = v19 | v20;
    *(_QWORD *)(v19 + 8) = v18 + 24;
    *v25 = *v25 & 7 | (v18 + 24);
  }
  sub_164B780(v18, v30);
  sub_12A86E0(v27, v18);
  sub_1B9A1B0((unsigned int *)(a1 + 280), a2, a3, v28, v21, v22);
  if ( *(_BYTE *)(v28 + 16) == 78 )
  {
    v24 = *(_QWORD *)(v28 - 24);
    if ( !*(_BYTE *)(v24 + 16) && (*(_BYTE *)(v24 + 33) & 0x20) != 0 && *(_DWORD *)(v24 + 36) == 4 )
      sub_14CE830(*(_QWORD *)(a1 + 64), v28);
  }
  if ( a4 )
    sub_14EF3D0(a1 + 384, &v28);
}
