// Function: sub_2A70140
// Address: 0x2a70140
//
void __fastcall sub_2A70140(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // rdx
  __int64 v7; // r15
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  unsigned int v12; // esi
  unsigned int v13; // eax
  unsigned __int8 v14; // dl
  unsigned int v15; // esi
  int v16; // eax
  unsigned __int8 *v17; // rax
  unsigned int v18; // eax
  __int64 *v19; // rax
  unsigned __int8 v20; // [rsp+18h] [rbp-168h]
  unsigned int v21; // [rsp+18h] [rbp-168h]
  int v23; // [rsp+28h] [rbp-158h]
  unsigned __int8 v24; // [rsp+30h] [rbp-150h]
  unsigned __int8 *v25; // [rsp+38h] [rbp-148h]
  __int64 v26; // [rsp+40h] [rbp-140h] BYREF
  unsigned int v27; // [rsp+48h] [rbp-138h]
  __int64 v28; // [rsp+50h] [rbp-130h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-128h]
  __int64 v30; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v31; // [rsp+68h] [rbp-118h]
  __int64 v32; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v33; // [rsp+78h] [rbp-108h]
  __int64 v34; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v35; // [rsp+88h] [rbp-F8h]
  __int64 v36; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v37; // [rsp+98h] [rbp-E8h]
  __int64 v38; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v39; // [rsp+A8h] [rbp-D8h]
  __int64 v40; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v41; // [rsp+B8h] [rbp-C8h]
  unsigned __int8 v42[8]; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+C8h] [rbp-B8h] BYREF
  unsigned int v44; // [rsp+D0h] [rbp-B0h]
  __int64 v45; // [rsp+D8h] [rbp-A8h] BYREF
  unsigned int v46; // [rsp+E0h] [rbp-A0h]
  unsigned __int8 v47[8]; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v48; // [rsp+F8h] [rbp-88h] BYREF
  unsigned int v49; // [rsp+100h] [rbp-80h]
  __int64 v50; // [rsp+108h] [rbp-78h] BYREF
  unsigned int v51; // [rsp+110h] [rbp-70h]
  __int64 v52[2]; // [rsp+120h] [rbp-60h] BYREF
  __int64 v53[10]; // [rsp+130h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v7 = *(_QWORD *)(a3 - 32 * v6);
  v25 = *(unsigned __int8 **)(a3 + 32 * (1 - v6));
  v8 = (unsigned __int8 *)sub_2A68BC0(a1, (unsigned __int8 *)v7);
  sub_22C05A0((__int64)v42, v8);
  v9 = (unsigned __int8 *)sub_2A68BC0(a1, v25);
  sub_22C05A0((__int64)v47, v9);
  sub_2A6FB70(a1, v7, a2);
  sub_2A6FB70(a1, (__int64)v25, a2);
  if ( v42[0] <= 1u || v47[0] <= 1u )
    goto LABEL_20;
  v10 = *(_QWORD *)(v7 + 8);
  if ( v42[0] == 4 )
    goto LABEL_23;
  v24 = v42[0];
  v11 = sub_BCB060(v10);
  v12 = v11;
  if ( v24 != 5 )
  {
    if ( v24 != 2 )
    {
LABEL_6:
      sub_AADB10((__int64)&v26, v12, 1);
      goto LABEL_7;
    }
    goto LABEL_33;
  }
  v12 = v11;
  if ( sub_9876C0(&v43) )
  {
LABEL_23:
    v27 = v44;
    if ( v44 > 0x40 )
      sub_C43780((__int64)&v26, (const void **)&v43);
    else
      v26 = v43;
    v29 = v46;
    if ( v46 > 0x40 )
      sub_C43780((__int64)&v28, (const void **)&v45);
    else
      v28 = v45;
  }
  else
  {
    if ( v42[0] == 2 )
    {
LABEL_33:
      sub_AD8380((__int64)&v26, v43);
      goto LABEL_7;
    }
    if ( v42[0] )
      goto LABEL_6;
    sub_AADB10((__int64)&v26, v12, 0);
  }
LABEL_7:
  v20 = v47[0];
  if ( v47[0] == 4
    || (v13 = sub_BCB060(v10), v14 = v20, v15 = v13, v20 == 5)
    && (v21 = v13, v19 = sub_9876C0(&v48), v14 = v47[0], v15 = v21, v19) )
  {
    v31 = v49;
    if ( v49 > 0x40 )
      sub_C43780((__int64)&v30, (const void **)&v48);
    else
      v30 = v48;
    v33 = v51;
    if ( v51 > 0x40 )
      sub_C43780((__int64)&v32, (const void **)&v50);
    else
      v32 = v50;
  }
  else if ( v14 == 2 )
  {
    sub_AD8380((__int64)&v30, v48);
  }
  else if ( v14 )
  {
    sub_AADB10((__int64)&v30, v15, 1);
  }
  else
  {
    sub_AADB10((__int64)&v30, v15, 0);
  }
  if ( !a4 )
  {
    v16 = sub_B5B5E0(a3);
    sub_ABCAA0((__int64)&v34, (__int64)&v26, v16, &v30);
    v39 = v35;
    if ( v35 > 0x40 )
      sub_C43780((__int64)&v38, (const void **)&v34);
    else
      v38 = v34;
    v41 = v37;
    if ( v37 > 0x40 )
      sub_C43780((__int64)&v40, (const void **)&v36);
    else
      v40 = v36;
    sub_22C06B0((__int64)v52, (__int64)&v38, 0);
    sub_2A689D0(a1, a2, (unsigned __int8 *)v52, 0x100000000LL);
    sub_22C0090((unsigned __int8 *)v52);
    sub_969240(&v40);
    sub_969240(&v38);
    sub_969240(&v36);
    sub_969240(&v34);
    goto LABEL_18;
  }
  v23 = sub_B5B690(a3);
  v18 = sub_B5B5E0(a3);
  sub_AB28E0((__int64)v52, v18, (__int64)&v30, v23);
  if ( !(unsigned __int8)sub_AB1BB0((__int64)v52, (__int64)&v26) )
  {
    sub_2A6A450(a1, a2);
    sub_969240(v53);
    sub_969240(v52);
LABEL_18:
    sub_969240(&v32);
    sub_969240(&v30);
    sub_969240(&v28);
    sub_969240(&v26);
    sub_22C0090(v47);
    sub_22C0090(v42);
    return;
  }
  v17 = (unsigned __int8 *)sub_AD6450(*(_QWORD *)(a2 + 8));
  sub_2A68820(a1, a2, v17);
  sub_969240(v53);
  sub_969240(v52);
  sub_969240(&v32);
  sub_969240(&v30);
  sub_969240(&v28);
  sub_969240(&v26);
LABEL_20:
  sub_22C0090(v47);
  sub_22C0090(v42);
}
