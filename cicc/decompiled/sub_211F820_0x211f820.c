// Function: sub_211F820
// Address: 0x211f820
//
__int64 __fastcall sub_211F820(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned __int8 *v7; // rax
  unsigned int v8; // r14d
  __int64 v9; // rax
  char v10; // di
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // esi
  bool v16; // cc
  char v17; // al
  const void **v18; // r11
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 *v22; // r13
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rsi
  __int64 v26; // r14
  const void **v28; // rdx
  unsigned int v29; // eax
  __int128 v30; // [rsp-10h] [rbp-80h]
  __int64 *v31; // [rsp+0h] [rbp-70h]
  __int64 v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  const void **v34; // [rsp+10h] [rbp-60h]
  char v35; // [rsp+1Eh] [rbp-52h]
  char v36; // [rsp+1Fh] [rbp-51h]
  __int64 v37; // [rsp+20h] [rbp-50h] BYREF
  __int64 v38; // [rsp+28h] [rbp-48h]
  const void **v39; // [rsp+30h] [rbp-40h]

  v7 = *(unsigned __int8 **)(a2 + 40);
  v36 = *v7;
  sub_1F40D10((__int64)&v37, *a1, *(_QWORD *)(a1[1] + 48), *v7, *((_QWORD *)v7 + 1));
  v35 = v38;
  v8 = (unsigned __int8)v38;
  v34 = v39;
  v31 = *(__int64 **)(a2 + 32);
  v9 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOBYTE(v37) = v10;
  v38 = v11;
  if ( v10 )
  {
    v12 = sub_211A7A0(v10);
    v13 = v31;
    v14 = a1[1];
    v15 = v12;
    v16 = v12 <= 0x20;
    if ( v12 != 32 )
      goto LABEL_3;
LABEL_20:
    v17 = 5;
    goto LABEL_6;
  }
  v29 = sub_1F58D40((__int64)&v37);
  v13 = v31;
  v14 = a1[1];
  v15 = v29;
  v16 = v29 <= 0x20;
  if ( v29 == 32 )
    goto LABEL_20;
LABEL_3:
  if ( v16 )
  {
    if ( v15 == 8 )
    {
      v17 = 3;
    }
    else
    {
      v17 = 4;
      if ( v15 != 16 )
      {
        v17 = 2;
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
    v17 = 6;
    goto LABEL_6;
  }
  if ( v15 == 128 )
  {
    v17 = 7;
    goto LABEL_6;
  }
LABEL_15:
  v17 = sub_1F58CC0(*(_QWORD **)(v14 + 48), v15);
  v14 = a1[1];
  v18 = v28;
  v13 = *(__int64 **)(a2 + 32);
LABEL_7:
  v19 = sub_1D32840((__int64 *)v14, v17, v18, *v13, v13[1], a3, a4, a5);
  v21 = *(_QWORD *)(a2 + 72);
  v22 = (__int64 *)a1[1];
  v23 = v19;
  v24 = v20;
  v37 = v21;
  if ( v21 )
  {
    v33 = v20;
    v32 = v19;
    sub_1623A60((__int64)&v37, v21, 2);
    v23 = v32;
    v24 = v33;
  }
  LODWORD(v38) = *(_DWORD *)(a2 + 64);
  if ( v36 == 8 )
  {
    v25 = 160;
  }
  else
  {
    if ( v35 != 8 )
      sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
    v25 = 161;
  }
  *((_QWORD *)&v30 + 1) = v24;
  *(_QWORD *)&v30 = v23;
  v26 = sub_1D309E0(v22, v25, (__int64)&v37, v8, v34, 0, a3, a4, a5, v30);
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
  return v26;
}
