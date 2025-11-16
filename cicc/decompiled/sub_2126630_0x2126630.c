// Function: sub_2126630
// Address: 0x2126630
//
__int64 __fastcall sub_2126630(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v5; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx
  char v9; // si
  __int64 v10; // rdx
  __int64 v11; // rdx
  char v12; // r15
  unsigned int v13; // eax
  __int64 v14; // r10
  unsigned int v15; // esi
  bool v16; // cc
  unsigned int v17; // eax
  const void **v18; // r8
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // r12
  const void **v25; // rdx
  unsigned int v26; // eax
  const void **v27; // [rsp+0h] [rbp-70h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int128 v29; // [rsp+10h] [rbp-60h]
  char v30[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v31; // [rsp+28h] [rbp-48h]
  __int64 v32; // [rsp+30h] [rbp-40h] BYREF
  int v33; // [rsp+38h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 40LL);
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v30[0] = v9;
  v31 = v10;
  *(_QWORD *)&v29 = sub_2125740(a1, *(_QWORD *)v7, *(_QWORD *)(v7 + 8));
  *((_QWORD *)&v29 + 1) = v11;
  v12 = **(_BYTE **)(v29 + 40);
  if ( v9 )
  {
    v13 = sub_211A7A0(v9);
    v14 = *(_QWORD *)(a1 + 8);
    v15 = v13;
    v16 = v13 <= 0x20;
    if ( v13 != 32 )
      goto LABEL_3;
LABEL_20:
    LOBYTE(v17) = 5;
    goto LABEL_6;
  }
  v26 = sub_1F58D40((__int64)v30);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = v26;
  v16 = v26 <= 0x20;
  if ( v26 == 32 )
    goto LABEL_20;
LABEL_3:
  if ( v16 )
  {
    if ( v15 == 8 )
    {
      LOBYTE(v17) = 3;
    }
    else
    {
      LOBYTE(v17) = 4;
      if ( v15 != 16 )
      {
        LOBYTE(v17) = 2;
        if ( v15 != 1 )
          goto LABEL_15;
      }
    }
LABEL_6:
    v18 = 0;
    goto LABEL_7;
  }
  if ( v15 == 64 )
  {
    LOBYTE(v17) = 6;
    goto LABEL_6;
  }
  if ( v15 == 128 )
  {
    LOBYTE(v17) = 7;
    goto LABEL_6;
  }
LABEL_15:
  v17 = sub_1F58CC0(*(_QWORD **)(v14 + 48), v15);
  v14 = *(_QWORD *)(a1 + 8);
  v5 = v17;
  v18 = v25;
LABEL_7:
  v19 = *(_QWORD *)(a2 + 72);
  LOBYTE(v5) = v17;
  v32 = v19;
  if ( v19 )
  {
    v27 = v18;
    v28 = v14;
    sub_1623A60((__int64)&v32, v19, 2);
    v18 = v27;
    v14 = v28;
  }
  v33 = *(_DWORD *)(a2 + 64);
  if ( v12 == 8 )
  {
    v20 = 160;
  }
  else
  {
    if ( v30[0] != 8 )
      sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
    v20 = 161;
  }
  v21 = sub_1D309E0((__int64 *)v14, v20, (__int64)&v32, v5, v18, 0, a3, a4, a5, v29);
  v23 = v22;
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  return sub_1D32840(
           *(__int64 **)(a1 + 8),
           **(unsigned __int8 **)(a2 + 40),
           *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
           v21,
           v23,
           a3,
           a4,
           a5);
}
