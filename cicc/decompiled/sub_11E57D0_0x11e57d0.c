// Function: sub_11E57D0
// Address: 0x11e57d0
//
__int64 __fastcall sub_11E57D0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 *v5; // rdi
  __int64 **v6; // r14
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int8 *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int8 *v13; // r10
  __int64 (__fastcall *v14)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned int v17; // r13d
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  _BYTE *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v25; // r13
  __int64 v26; // rbx
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rax
  unsigned __int8 *v30; // [rsp+0h] [rbp-C0h]
  unsigned __int8 *v31; // [rsp+0h] [rbp-C0h]
  __int64 v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  __int64 v34; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-98h]
  _QWORD v37[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v38; // [rsp+50h] [rbp-70h]
  _QWORD v39[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v40; // [rsp+80h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4);
  v5 = *(__int64 **)(a3 + 72);
  BYTE4(v36) = 0;
  v6 = *(__int64 ***)(a2 + 8);
  v7 = *(_QWORD *)(a2 - 32LL * (v4 & 0x7FFFFFF));
  v8 = *(_QWORD *)(v7 + 8);
  v33 = v7;
  v39[0] = "cttz";
  v34 = v8;
  v40 = 259;
  v37[0] = v7;
  v35 = v8;
  v37[1] = sub_ACD6D0(v5);
  v9 = sub_B33D10(a3, 0x43u, (__int64)&v35, 1, (int)v37, 2, v36, (__int64)v39);
  v38 = 257;
  v10 = (unsigned __int8 *)v9;
  v11 = sub_AD64C0(*(_QWORD *)(v9 + 8), 1, 0);
  v12 = *(_QWORD *)(a3 + 80);
  v13 = (unsigned __int8 *)v11;
  v14 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v12 + 32LL);
  if ( v14 != sub_9201A0 )
  {
    v31 = v13;
    v29 = v14(v12, 13u, v10, v13, 0, 0);
    v13 = v31;
    v16 = v29;
    goto LABEL_7;
  }
  if ( *v10 <= 0x15u && *v13 <= 0x15u )
  {
    v30 = v13;
    if ( (unsigned __int8)sub_AC47B0(13) )
      v15 = sub_AD5570(13, (__int64)v10, v30, 0, 0);
    else
      v15 = sub_AABE40(0xDu, v10, v30);
    v13 = v30;
    v16 = v15;
LABEL_7:
    if ( v16 )
      goto LABEL_8;
  }
  v40 = 257;
  v16 = sub_B504D0(13, (__int64)v10, (__int64)v13, (__int64)v39, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v16,
    v37,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
  {
    v25 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    v26 = *(_QWORD *)a3;
    do
    {
      v27 = *(_QWORD *)(v26 + 8);
      v28 = *(_DWORD *)v26;
      v26 += 16;
      sub_B99FD0(v16, v28, v27);
    }
    while ( v25 != v26 );
  }
LABEL_8:
  v40 = 257;
  v17 = sub_BCB060(*(_QWORD *)(v16 + 8));
  v18 = (unsigned int)(v17 <= (unsigned int)sub_BCB060((__int64)v6)) + 38;
  v19 = sub_11DB4B0((__int64 *)a3, v18, v16, v6, (__int64)v39, 0, v37[0], 0);
  v40 = 257;
  v32 = v19;
  v20 = (_BYTE *)sub_AD6530(v34, v18);
  v21 = sub_92B530((unsigned int **)a3, 0x21u, v33, v20, (__int64)v39);
  v40 = 257;
  v22 = v21;
  v23 = sub_AD64C0((__int64)v6, 0, 0);
  return sub_B36550((unsigned int **)a3, v22, v32, v23, (__int64)v39, 0);
}
