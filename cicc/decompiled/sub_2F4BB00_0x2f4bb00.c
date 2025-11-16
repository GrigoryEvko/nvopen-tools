// Function: sub_2F4BB00
// Address: 0x2f4bb00
//
__int64 __fastcall sub_2F4BB00(__int64 a1, __int64 a2, _QWORD *a3)
{
  void (__fastcall *v5)(__int64 *, __int64, __int64); // rax
  char v6; // r14
  void (__fastcall *v7)(__int64 *, __int64 *, __int64); // rax
  __int64 **v9; // rdx
  __int64 v10; // rcx
  void *v11; // r9
  void **v12; // rax
  __int64 **v13; // rsi
  __int64 *v14; // rax
  void *v15; // [rsp+0h] [rbp-5B0h]
  __int64 v16; // [rsp+10h] [rbp-5A0h] BYREF
  void **v17; // [rsp+18h] [rbp-598h]
  void (__fastcall *v18)(__int64 *, __int64 *, __int64); // [rsp+20h] [rbp-590h]
  __int64 v19; // [rsp+28h] [rbp-588h]
  _BYTE v20[16]; // [rsp+30h] [rbp-580h] BYREF
  _BYTE v21[8]; // [rsp+40h] [rbp-570h] BYREF
  unsigned __int64 v22; // [rsp+48h] [rbp-568h]
  int v23; // [rsp+54h] [rbp-55Ch]
  int v24; // [rsp+58h] [rbp-558h]
  char v25; // [rsp+5Ch] [rbp-554h]
  _BYTE v26[16]; // [rsp+60h] [rbp-550h] BYREF
  _QWORD v27[4]; // [rsp+70h] [rbp-540h] BYREF
  _BYTE v28[320]; // [rsp+90h] [rbp-520h] BYREF
  _BYTE v29[16]; // [rsp+1D0h] [rbp-3E0h] BYREF
  void (__fastcall *v30)(__int64 *, __int64 *, __int64); // [rsp+1E0h] [rbp-3D0h]
  __int64 v31; // [rsp+1E8h] [rbp-3C8h]
  __int64 v32; // [rsp+1F0h] [rbp-3C0h]
  int *v33; // [rsp+1F8h] [rbp-3B8h]
  __int64 v34; // [rsp+200h] [rbp-3B0h]
  int v35; // [rsp+208h] [rbp-3A8h] BYREF
  char *v36; // [rsp+210h] [rbp-3A0h]
  __int64 v37; // [rsp+218h] [rbp-398h]
  char v38; // [rsp+220h] [rbp-390h] BYREF
  __int64 v39; // [rsp+2E0h] [rbp-2D0h]
  int v40; // [rsp+2E8h] [rbp-2C8h]
  __int64 v41; // [rsp+2F0h] [rbp-2C0h]
  __int64 v42; // [rsp+2F8h] [rbp-2B8h]
  __int64 v43; // [rsp+300h] [rbp-2B0h]
  int v44; // [rsp+308h] [rbp-2A8h]
  __int64 v45; // [rsp+310h] [rbp-2A0h]
  __int64 v46; // [rsp+318h] [rbp-298h]
  __int64 v47; // [rsp+320h] [rbp-290h]
  int v48; // [rsp+328h] [rbp-288h]
  __int64 v49; // [rsp+330h] [rbp-280h]
  __int64 v50; // [rsp+338h] [rbp-278h]
  __int64 v51; // [rsp+340h] [rbp-270h]
  int v52; // [rsp+348h] [rbp-268h]
  char *v53; // [rsp+350h] [rbp-260h]
  __int64 v54; // [rsp+358h] [rbp-258h]
  char v55; // [rsp+360h] [rbp-250h] BYREF
  int v56; // [rsp+390h] [rbp-220h]
  __int64 v57; // [rsp+398h] [rbp-218h]
  __int64 v58; // [rsp+3A0h] [rbp-210h]
  __int64 v59; // [rsp+3A8h] [rbp-208h]
  char *v60; // [rsp+3B0h] [rbp-200h]
  __int64 v61; // [rsp+3B8h] [rbp-1F8h]
  char v62; // [rsp+3C0h] [rbp-1F0h] BYREF
  _QWORD *v63; // [rsp+4C8h] [rbp-E8h]
  __int64 v64; // [rsp+4D0h] [rbp-E0h]
  _QWORD v65[2]; // [rsp+4D8h] [rbp-D8h] BYREF
  char v66; // [rsp+4E8h] [rbp-C8h] BYREF
  _BYTE *v67; // [rsp+508h] [rbp-A8h]
  __int64 v68; // [rsp+510h] [rbp-A0h]
  _BYTE v69[56]; // [rsp+518h] [rbp-98h] BYREF
  __int64 v70; // [rsp+550h] [rbp-60h]
  __int64 v71; // [rsp+558h] [rbp-58h]
  __int64 v72; // [rsp+560h] [rbp-50h]
  __int64 v73; // [rsp+568h] [rbp-48h]
  int v74; // [rsp+570h] [rbp-40h]
  char v75; // [rsp+578h] [rbp-38h]

  a3[43] &= 0xFFEuLL;
  v5 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a2 + 16);
  v18 = 0;
  v6 = *(_BYTE *)(a2 + 48);
  if ( v5 )
  {
    v5(&v16, a2, 2);
    v19 = *(_QWORD *)(a2 + 24);
    v18 = *(void (__fastcall **)(__int64 *, __int64 *, __int64))(a2 + 16);
  }
  memset(v27, 0, sizeof(v27));
  sub_2F5FEE0(v28);
  v7 = v18;
  v30 = 0;
  if ( v18 )
  {
    v18((__int64 *)v29, &v16, 2);
    v31 = v19;
    v7 = v18;
    v30 = v18;
  }
  v33 = &v35;
  v36 = &v38;
  v60 = &v62;
  v37 = 0x800000000LL;
  v53 = &v55;
  v63 = v65;
  v65[1] = 0x800000000LL;
  v32 = 0;
  v34 = 0;
  v35 = -1;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v54 = 0x600000000LL;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v61 = 0x2000000000LL;
  v64 = 0;
  v65[0] = &v66;
  v67 = v69;
  v68 = 0x600000000LL;
  v69[48] = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = v6;
  if ( v7 )
    v7(&v16, &v16, 3);
  if ( !(unsigned __int8)sub_2F49070(v27, a3) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_9;
  }
  sub_2EAFFB0((__int64)&v16);
  v11 = (void *)(a1 + 32);
  if ( v23 == v24 )
  {
    if ( BYTE4(v19) )
    {
      v12 = v17;
      v13 = (__int64 **)&v17[HIDWORD(v18)];
      v10 = HIDWORD(v18);
      v9 = (__int64 **)v17;
      if ( v17 != (void **)v13 )
      {
        while ( *v9 != &qword_4F82400 )
        {
          if ( v13 == ++v9 )
          {
LABEL_15:
            while ( *v12 != &unk_4F82408 )
            {
              if ( ++v12 == (void **)v9 )
                goto LABEL_20;
            }
            goto LABEL_16;
          }
        }
        goto LABEL_16;
      }
      goto LABEL_20;
    }
    v14 = sub_C8CA60((__int64)&v16, (__int64)&qword_4F82400);
    v11 = (void *)(a1 + 32);
    if ( v14 )
      goto LABEL_16;
  }
  if ( !BYTE4(v19) )
  {
LABEL_22:
    v15 = v11;
    sub_C8CC70((__int64)&v16, (__int64)&unk_4F82408, (__int64)v9, v10, (__int64)&v16, (__int64)v11);
    v11 = v15;
    goto LABEL_16;
  }
  v12 = v17;
  v10 = HIDWORD(v18);
  v9 = (__int64 **)&v17[HIDWORD(v18)];
  if ( v9 != (__int64 **)v17 )
    goto LABEL_15;
LABEL_20:
  if ( (unsigned int)v18 <= (unsigned int)v10 )
    goto LABEL_22;
  HIDWORD(v18) = v10 + 1;
  *v9 = (__int64 *)&unk_4F82408;
  ++v16;
LABEL_16:
  sub_C8CF70(a1, v11, 2, (__int64)v20, (__int64)&v16);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v26, (__int64)v21);
  if ( !v25 )
    _libc_free(v22);
  if ( !BYTE4(v19) )
    _libc_free((unsigned __int64)v17);
LABEL_9:
  sub_2F43140((__int64)v27);
  a3[43] |= 8LL * (*(_BYTE *)(a2 + 48) != 0);
  return a1;
}
