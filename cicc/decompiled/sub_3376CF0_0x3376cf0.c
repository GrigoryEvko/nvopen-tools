// Function: sub_3376CF0
// Address: 0x3376cf0
//
void __fastcall sub_3376CF0(
        __int64 a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        char a9)
{
  char v13; // al
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // r8d
  __int64 v25; // rax
  __int64 v26; // rax
  char *v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rax
  char *v30; // rdx
  __int64 v31; // [rsp+0h] [rbp-B0h]
  unsigned int v32; // [rsp+0h] [rbp-B0h]
  __int64 v34; // [rsp+10h] [rbp-A0h] BYREF
  int v35; // [rsp+18h] [rbp-98h]
  __int64 v36; // [rsp+20h] [rbp-90h] BYREF
  char *v37; // [rsp+28h] [rbp-88h]
  __int64 v38; // [rsp+30h] [rbp-80h]
  __int64 v39; // [rsp+38h] [rbp-78h]
  __int64 v40; // [rsp+40h] [rbp-70h]
  __int64 v41; // [rsp+48h] [rbp-68h]
  __int64 v42; // [rsp+50h] [rbp-60h]
  __int64 v43; // [rsp+58h] [rbp-58h] BYREF
  int v44; // [rsp+60h] [rbp-50h]
  __int64 v45; // [rsp+68h] [rbp-48h] BYREF
  int v46; // [rsp+70h] [rbp-40h]
  int v47; // [rsp+74h] [rbp-3Ch]
  char v48; // [rsp+78h] [rbp-38h]

  v13 = *a2;
  if ( (unsigned __int8)(*a2 - 82) > 1u )
    goto LABEL_4;
  if ( a5 == a6 )
    goto LABEL_26;
  v31 = *(_QWORD *)(a5 + 16);
  if ( !(unsigned __int8)sub_3373C30(a1, *((_QWORD *)a2 - 8), v31)
    || !(unsigned __int8)sub_3373C30(a1, *((_QWORD *)a2 - 4), v31) )
  {
LABEL_4:
    v14 = *(_QWORD *)a1;
    v15 = *(_DWORD *)(a1 + 848);
    v34 = 0;
    v35 = v15;
    if ( v14 )
    {
      if ( &v34 != (__int64 *)(v14 + 48) )
      {
        v16 = *(_QWORD *)(v14 + 48);
        v34 = v16;
        if ( v16 )
          sub_B96E90((__int64)&v34, v16, 1);
      }
    }
    v17 = sub_ACD6D0(*(__int64 **)(*(_QWORD *)(a1 + 864) + 64LL));
    LODWORD(v36) = a9 == 0 ? 17 : 22;
    v39 = v17;
    v37 = a2;
    v38 = 0;
    v40 = a3;
    v41 = a4;
    v42 = a5;
    v43 = v34;
    if ( v34 )
    {
      sub_B96E90((__int64)&v43, v34, 1);
      v18 = v34;
      v45 = 0;
      v48 = 0;
      v44 = v35;
      v46 = a7;
      v47 = a8;
      if ( !v34 )
        goto LABEL_11;
      goto LABEL_10;
    }
    goto LABEL_34;
  }
  v13 = *a2;
LABEL_26:
  v23 = *((_WORD *)a2 + 1) & 0x3F;
  if ( v13 == 82 )
  {
    if ( a9 )
      v23 = (unsigned int)sub_B52870(v23);
    v24 = sub_34B9220(v23);
  }
  else
  {
    if ( a9 )
      v23 = (unsigned int)sub_B52870(v23);
    v24 = sub_34B9180(v23);
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 856) + 864LL) & 4) != 0 )
      v24 = sub_34B9190(v24);
  }
  v25 = *(_QWORD *)a1;
  v35 = *(_DWORD *)(a1 + 848);
  if ( !v25 || &v34 == (__int64 *)(v25 + 48) || (v28 = *(_QWORD *)(v25 + 48), (v34 = v28) == 0) )
  {
    v26 = *((_QWORD *)a2 - 4);
    v27 = (char *)*((_QWORD *)a2 - 8);
    v40 = a3;
    LODWORD(v36) = v24;
    v39 = v26;
    v37 = v27;
    v38 = 0;
    v41 = a4;
    v42 = a5;
    v43 = 0;
LABEL_34:
    v45 = 0;
    v48 = 0;
    v44 = v35;
    v46 = a7;
    v47 = a8;
    goto LABEL_11;
  }
  v32 = v24;
  sub_B96E90((__int64)&v34, v28, 1);
  v29 = *((_QWORD *)a2 - 4);
  v30 = (char *)*((_QWORD *)a2 - 8);
  v38 = 0;
  v40 = a3;
  v39 = v29;
  LODWORD(v36) = v32;
  v37 = v30;
  v41 = a4;
  v42 = a5;
  v43 = v34;
  if ( !v34 )
    goto LABEL_34;
  sub_B96E90((__int64)&v43, v34, 1);
  v18 = v34;
  v45 = 0;
  v48 = 0;
  v44 = v35;
  v46 = a7;
  v47 = a8;
  if ( v34 )
LABEL_10:
    sub_B91220((__int64)&v34, v18);
LABEL_11:
  v19 = *(_QWORD **)(a1 + 896);
  v20 = v19[2];
  if ( v20 == v19[3] )
  {
    sub_3376950(v19 + 1, v19[2], (__int64)&v36);
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = v36;
      *(_QWORD *)(v20 + 8) = v37;
      *(_QWORD *)(v20 + 16) = v38;
      *(_QWORD *)(v20 + 24) = v39;
      *(_QWORD *)(v20 + 32) = v40;
      *(_QWORD *)(v20 + 40) = v41;
      *(_QWORD *)(v20 + 48) = v42;
      v21 = v43;
      *(_QWORD *)(v20 + 56) = v43;
      if ( v21 )
        sub_B96E90(v20 + 56, v21, 1);
      *(_DWORD *)(v20 + 64) = v44;
      v22 = v45;
      *(_QWORD *)(v20 + 72) = v45;
      if ( v22 )
        sub_B96E90(v20 + 72, v22, 1);
      *(_DWORD *)(v20 + 80) = v46;
      *(_DWORD *)(v20 + 84) = v47;
      *(_BYTE *)(v20 + 88) = v48;
      v20 = v19[2];
    }
    v19[2] = v20 + 96;
  }
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
}
