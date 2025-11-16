// Function: sub_2126D00
// Address: 0x2126d00
//
__int64 __fastcall sub_2126D00(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v5; // r15d
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rax
  unsigned int v18; // eax
  char v19; // dl
  __int64 v20; // rdi
  unsigned int v21; // esi
  bool v22; // cc
  unsigned int v23; // eax
  const void **v24; // r8
  __int64 v25; // rsi
  unsigned int v26; // edx
  __int64 v27; // r12
  const void **v29; // rdx
  unsigned int v30; // eax
  __int128 v31; // [rsp-10h] [rbp-80h]
  char v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+20h] [rbp-50h] BYREF
  int v35; // [rsp+28h] [rbp-48h]
  char v36[8]; // [rsp+30h] [rbp-40h] BYREF
  __int64 v37; // [rsp+38h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *(_QWORD *)(v8 + 40);
  v11 = *(_QWORD *)(v8 + 48);
  v34 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v34, v9, 2);
  v35 = *(_DWORD *)(a2 + 64);
  v12 = sub_2125740(a1, v10, v11);
  v14 = v13;
  v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL);
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v36[0] = v16;
  v37 = v17;
  if ( v16 )
  {
    v32 = v16;
    v18 = sub_211A7A0(v16);
    v19 = v32;
    v20 = *(_QWORD *)(a1 + 8);
    v21 = v18;
    v22 = v18 <= 0x20;
    if ( v18 != 32 )
      goto LABEL_5;
LABEL_20:
    LOBYTE(v23) = 5;
    goto LABEL_8;
  }
  v30 = sub_1F58D40((__int64)v36);
  v19 = 0;
  v20 = *(_QWORD *)(a1 + 8);
  v21 = v30;
  v22 = v30 <= 0x20;
  if ( v30 == 32 )
    goto LABEL_20;
LABEL_5:
  if ( v22 )
  {
    if ( v21 == 8 )
    {
      LOBYTE(v23) = 3;
    }
    else
    {
      LOBYTE(v23) = 4;
      if ( v21 != 16 )
      {
        LOBYTE(v23) = 2;
        if ( v21 != 1 )
          goto LABEL_18;
      }
    }
LABEL_8:
    v24 = 0;
    goto LABEL_9;
  }
  if ( v21 == 64 )
  {
    LOBYTE(v23) = 6;
    goto LABEL_8;
  }
  if ( v21 == 128 )
  {
    LOBYTE(v23) = 7;
    goto LABEL_8;
  }
LABEL_18:
  v23 = sub_1F58CC0(*(_QWORD **)(v20 + 48), v21);
  v20 = *(_QWORD *)(a1 + 8);
  v24 = v29;
  v5 = v23;
  v19 = v36[0];
LABEL_9:
  LOBYTE(v5) = v23;
  if ( *(_BYTE *)(*(_QWORD *)(v12 + 40) + 16LL * (unsigned int)v14) == 8 )
  {
    v25 = 160;
  }
  else
  {
    if ( v19 != 8 )
      sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
    v25 = 161;
  }
  *((_QWORD *)&v31 + 1) = v14;
  *(_QWORD *)&v31 = v12;
  v33 = sub_1D309E0((__int64 *)v20, v25, (__int64)&v34, v5, v24, 0, a3, a4, a5, v31);
  v27 = sub_1D2BB40(
          *(_QWORD **)(a1 + 8),
          **(_QWORD **)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
          (__int64)&v34,
          v33,
          v26,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL),
          *(_QWORD *)(a2 + 104));
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  return v27;
}
