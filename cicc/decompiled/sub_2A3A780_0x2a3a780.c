// Function: sub_2A3A780
// Address: 0x2a3a780
//
__int64 __fastcall sub_2A3A780(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 **v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rax
  _BYTE *v12; // r14
  __int64 v13; // rdi
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v15; // r13
  __int64 v17; // rdx
  int v18; // r14d
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // [rsp+8h] [rbp-D8h] BYREF
  __int64 v24; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+18h] [rbp-C8h]
  char v26[32]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-A0h]
  char v28[32]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v29; // [rsp+70h] [rbp-70h]
  char v30[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v31; // [rsp+A0h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 48);
  v3 = *(_QWORD *)(a1 + 72);
  v29 = 257;
  v4 = *(_QWORD *)(*(_QWORD *)(v2 + 72) + 40LL);
  v5 = sub_AE4420(v4 + 312, v3, 0);
  v6 = *(_QWORD **)(a1 + 72);
  HIDWORD(v25) = 0;
  v27 = 257;
  v7 = (__int64 **)v5;
  v8 = sub_BCB2D0(v6);
  v9 = sub_AD6530(v8, v3);
  LODWORD(v3) = *(_DWORD *)(v4 + 316);
  v10 = *(__int64 **)(a1 + 72);
  v24 = v9;
  v23 = sub_BCE3C0(v10, v3);
  v11 = sub_B33D10(a1, 0xB2u, (__int64)&v23, 1, (int)&v24, 1, v25, (__int64)v26);
  v12 = (_BYTE *)v11;
  if ( v7 == *(__int64 ***)(v11 + 8) )
    return v11;
  v13 = *(_QWORD *)(a1 + 80);
  v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v13 + 120LL);
  if ( v14 != sub_920130 )
  {
    v15 = v14(v13, 47u, v12, (__int64)v7);
    goto LABEL_6;
  }
  if ( *v12 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x2Fu) )
      v15 = sub_ADAB70(47, (unsigned __int64)v12, v7, 0);
    else
      v15 = sub_AA93C0(0x2Fu, (unsigned __int64)v12, (__int64)v7);
LABEL_6:
    if ( v15 )
      return v15;
  }
  v31 = 257;
  v15 = sub_B51D30(47, (__int64)v12, (__int64)v7, (__int64)v30, 0, 0);
  if ( (unsigned __int8)sub_920620(v15) )
  {
    v17 = *(_QWORD *)(a1 + 96);
    v18 = *(_DWORD *)(a1 + 104);
    if ( v17 )
      sub_B99FD0(v15, 3u, v17);
    sub_B45150(v15, v18);
  }
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v15,
    v28,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v19 = *(_QWORD *)a1;
  v20 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  while ( v20 != v19 )
  {
    v21 = *(_QWORD *)(v19 + 8);
    v22 = *(_DWORD *)v19;
    v19 += 16;
    sub_B99FD0(v15, v22, v21);
  }
  return v15;
}
