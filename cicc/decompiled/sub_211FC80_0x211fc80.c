// Function: sub_211FC80
// Address: 0x211fc80
//
__int64 __fastcall sub_211FC80(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v5; // r14d
  __int64 v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned __int8 *v13; // rax
  char *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rsi
  unsigned int v17; // r15d
  unsigned int v18; // eax
  char v19; // dl
  __int64 v20; // rdi
  unsigned int v21; // esi
  bool v22; // cc
  char v23; // al
  const void **v24; // r8
  __int64 v25; // rsi
  __int128 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // r12
  unsigned int v30; // eax
  const void **v31; // rdx
  unsigned int v32; // eax
  __int128 v33; // [rsp-10h] [rbp-A0h]
  char v34; // [rsp+Eh] [rbp-82h]
  unsigned __int8 v35; // [rsp+Fh] [rbp-81h]
  const void **v36; // [rsp+10h] [rbp-80h]
  char v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+20h] [rbp-70h] BYREF
  int v39; // [rsp+28h] [rbp-68h]
  char v40[8]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v41; // [rsp+38h] [rbp-58h]
  _BYTE v42[8]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int8 v43; // [rsp+48h] [rbp-48h]
  const void **v44; // [rsp+50h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 72);
  v38 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v38, v8, 2);
  v39 = *(_DWORD *)(a2 + 64);
  v9 = *(__int64 **)(a2 + 32);
  v10 = *v9;
  v11 = *v9;
  v12 = v9[1];
  v13 = *(unsigned __int8 **)(a2 + 40);
  v14 = *(char **)(v10 + 40);
  v15 = *((_QWORD *)v13 + 1);
  v40[0] = *v13;
  v16 = *(_QWORD *)a1;
  v41 = v15;
  v37 = *v14;
  sub_1F40D10((__int64)v42, v16, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v13, *((_QWORD *)v13 + 1));
  v35 = v43;
  v17 = v43;
  v36 = v44;
  if ( v40[0] )
  {
    v34 = v40[0];
    v18 = sub_211A7A0(v40[0]);
    v19 = v34;
    v20 = *(_QWORD *)(a1 + 8);
    v21 = v18;
    v22 = v18 <= 0x20;
    if ( v18 != 32 )
      goto LABEL_5;
LABEL_23:
    v23 = 5;
    goto LABEL_8;
  }
  v32 = sub_1F58D40((__int64)v40);
  v19 = 0;
  v20 = *(_QWORD *)(a1 + 8);
  v21 = v32;
  v22 = v32 <= 0x20;
  if ( v32 == 32 )
    goto LABEL_23;
LABEL_5:
  if ( v22 )
  {
    if ( v21 == 8 )
    {
      v23 = 3;
    }
    else
    {
      v23 = 4;
      if ( v21 != 16 )
      {
        v23 = 2;
        if ( v21 != 1 )
        {
LABEL_19:
          v30 = sub_1F58CC0(*(_QWORD **)(v20 + 48), v21);
          v20 = *(_QWORD *)(a1 + 8);
          v5 = v30;
          v24 = v31;
          v19 = v40[0];
          LOBYTE(v5) = v30;
          if ( v37 != 8 )
            goto LABEL_20;
LABEL_9:
          v25 = 160;
          goto LABEL_10;
        }
      }
    }
  }
  else if ( v21 == 64 )
  {
    v23 = 6;
  }
  else
  {
    if ( v21 != 128 )
      goto LABEL_19;
    v23 = 7;
  }
LABEL_8:
  v24 = 0;
  LOBYTE(v5) = v23;
  if ( v37 == 8 )
    goto LABEL_9;
LABEL_20:
  if ( v19 != 8 )
    goto LABEL_24;
  v25 = 161;
LABEL_10:
  *((_QWORD *)&v33 + 1) = v12;
  *(_QWORD *)&v33 = v11;
  *(_QWORD *)&v26 = sub_1D309E0((__int64 *)v20, v25, (__int64)&v38, v5, v24, 0, a3, a4, a5, v33);
  if ( v40[0] != 8 )
  {
    v27 = 161;
    if ( v35 == 8 )
      goto LABEL_12;
LABEL_24:
    sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
  }
  v27 = 160;
LABEL_12:
  v28 = sub_1D309E0(*(__int64 **)(a1 + 8), v27, (__int64)&v38, v17, v36, 0, a3, a4, a5, v26);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v28;
}
