// Function: sub_200D2A0
// Address: 0x200d2a0
//
__int64 __fastcall sub_200D2A0(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  unsigned int v6; // r15d
  __int64 v11; // rax
  char v12; // di
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // r10
  unsigned int v17; // esi
  bool v18; // cc
  unsigned int v19; // eax
  const void **v20; // r8
  __int64 v21; // rsi
  __int64 v22; // r12
  const void **v24; // rdx
  __int128 v25; // [rsp-10h] [rbp-60h]
  const void **v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h] BYREF
  __int64 v31; // [rsp+18h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v12 = *(_BYTE *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LOBYTE(v30) = v12;
  v31 = v13;
  if ( v12 )
  {
    v14 = sub_200D0E0(v12);
    v16 = *(_QWORD *)(v15 + 8);
    v17 = v14;
    v18 = v14 <= 0x20;
    if ( v14 != 32 )
      goto LABEL_3;
LABEL_18:
    LOBYTE(v19) = 5;
    goto LABEL_6;
  }
  v29 = a1;
  v17 = sub_1F58D40((__int64)&v30);
  v16 = *(_QWORD *)(v29 + 8);
  v18 = v17 <= 0x20;
  if ( v17 == 32 )
    goto LABEL_18;
LABEL_3:
  if ( v18 )
  {
    if ( v17 == 8 )
    {
      LOBYTE(v19) = 3;
    }
    else
    {
      LOBYTE(v19) = 4;
      if ( v17 != 16 )
      {
        LOBYTE(v19) = 2;
        if ( v17 != 1 )
          goto LABEL_13;
      }
    }
LABEL_6:
    v20 = 0;
    goto LABEL_7;
  }
  if ( v17 == 64 )
  {
    LOBYTE(v19) = 6;
    goto LABEL_6;
  }
  if ( v17 == 128 )
  {
    LOBYTE(v19) = 7;
    goto LABEL_6;
  }
LABEL_13:
  v28 = v16;
  v19 = sub_1F58CC0(*(_QWORD **)(v16 + 48), v17);
  v16 = v28;
  v6 = v19;
  v20 = v24;
LABEL_7:
  v21 = *(_QWORD *)(a2 + 72);
  LOBYTE(v6) = v19;
  v30 = v21;
  if ( v21 )
  {
    v26 = v20;
    v27 = v16;
    sub_1623A60((__int64)&v30, v21, 2);
    v20 = v26;
    v16 = v27;
  }
  *((_QWORD *)&v25 + 1) = a3;
  *(_QWORD *)&v25 = a2;
  LODWORD(v31) = *(_DWORD *)(a2 + 64);
  v22 = sub_1D309E0((__int64 *)v16, 158, (__int64)&v30, v6, v20, 0, a4, a5, a6, v25);
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  return v22;
}
