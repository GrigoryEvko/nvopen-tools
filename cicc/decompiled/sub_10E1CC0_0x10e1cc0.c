// Function: sub_10E1CC0
// Address: 0x10e1cc0
//
__int64 __fastcall sub_10E1CC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rcx
  unsigned __int64 v6; // rax
  unsigned __int8 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 *v12; // r15
  __int64 v13; // r10
  _QWORD *v14; // rax
  __int64 v15; // r9
  __int64 v16; // r13
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rdx
  unsigned int v21; // esi
  unsigned int **v22; // rdi
  __int64 v23; // rcx
  __int64 v25; // r11
  __int64 *v26; // r10
  _QWORD *v27; // rax
  __int64 v28; // r9
  __int64 *v29; // r10
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // [rsp-10h] [rbp-C0h]
  __int64 v35; // [rsp+8h] [rbp-A8h]
  __int64 v36; // [rsp+10h] [rbp-A0h]
  __int64 *v37; // [rsp+18h] [rbp-98h]
  __int64 *v38; // [rsp+18h] [rbp-98h]
  _QWORD v39[4]; // [rsp+20h] [rbp-90h] BYREF
  char v40; // [rsp+40h] [rbp-70h]
  char v41; // [rsp+41h] [rbp-6Fh]
  _BYTE v42[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v43; // [rsp+70h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(a2 - 32 * v3);
  v5 = *(_QWORD *)(a2 + 32 * (1 - v3));
  v6 = *(_QWORD *)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = *(_QWORD *)v6;
  v7 = 0;
  if ( v6 )
  {
    _BitScanReverse64(&v6, v6);
    v7 = 63 - (v6 ^ 0x3F);
  }
  if ( (unsigned __int8)sub_9BA060(*(unsigned __int8 **)(a2 + 32 * (2 - v3))) )
  {
    v25 = *(_QWORD *)(a2 + 8);
    v26 = *(__int64 **)(a1 + 32);
    v39[0] = "unmaskedload";
    v36 = v25;
    v37 = v26;
    v41 = 1;
    v40 = 3;
    v43 = 257;
    v27 = sub_BD2C40(80, unk_3F10A14);
    v29 = v37;
    v11 = (__int64)v27;
    if ( v27 )
    {
      sub_B4D190((__int64)v27, v36, v4, (__int64)v42, 0, v7, 0, 0);
      v28 = v34;
      v29 = v37;
    }
    v38 = v29;
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64, __int64))(*(_QWORD *)v29[11] + 16LL))(
      v29[11],
      v11,
      v39,
      v29[7],
      v29[8],
      v28);
    v30 = *v38;
    v31 = *v38 + 16LL * *((unsigned int *)v38 + 2);
    if ( *v38 != v31 )
    {
      do
      {
        v32 = *(_QWORD *)(v30 + 8);
        v33 = *(_DWORD *)v30;
        v30 += 16;
        sub_B99FD0(v11, v33, v32);
      }
      while ( v31 != v30 );
    }
    sub_B47C00(v11, a2, 0, 0);
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 64);
    v9 = sub_B43CC0(a2);
    v10 = v8;
    v11 = 0;
    if ( sub_D30730(v4, *(_QWORD *)(a2 + 8), v9, a2, v10, 0, 0) )
    {
      v41 = 1;
      v12 = *(__int64 **)(a1 + 32);
      v13 = *(_QWORD *)(a2 + 8);
      v43 = 257;
      v39[0] = "unmaskedload";
      v35 = v13;
      v40 = 3;
      v14 = sub_BD2C40(80, unk_3F10A14);
      v15 = v7;
      v16 = (__int64)v14;
      if ( v14 )
        sub_B4D190((__int64)v14, v35, v4, (__int64)v42, 0, v15, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64, __int64))(*(_QWORD *)v12[11] + 16LL))(
        v12[11],
        v16,
        v39,
        v12[7],
        v12[8],
        v15);
      v17 = 16LL * *((unsigned int *)v12 + 2);
      v18 = *v12;
      v19 = v18 + v17;
      while ( v19 != v18 )
      {
        v20 = *(_QWORD *)(v18 + 8);
        v21 = *(_DWORD *)v18;
        v18 += 16;
        sub_B99FD0(v16, v21, v20);
      }
      sub_B47C00(v16, a2, 0, 0);
      v22 = *(unsigned int ***)(a1 + 32);
      v23 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v43 = 257;
      return sub_B36550(v22, *(_QWORD *)(a2 + 32 * (2 - v23)), v16, *(_QWORD *)(a2 + 32 * (3 - v23)), (__int64)v42, 0);
    }
  }
  return v11;
}
