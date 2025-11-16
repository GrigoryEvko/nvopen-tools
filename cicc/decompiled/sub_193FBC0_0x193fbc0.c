// Function: sub_193FBC0
// Address: 0x193fbc0
//
__int64 __fastcall sub_193FBC0(__int64 a1, __int64 a2, __int64 **a3, char a4, __int64 a5)
{
  __int64 v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rdi
  int v16; // edx
  unsigned int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rbx
  unsigned __int8 *v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int8 *v24; // rsi
  int v25; // eax
  __int64 **v26; // rax
  __int64 v27; // rax
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  int v33; // edi
  __int64 *v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  int v39; // r9d
  unsigned __int8 *v41; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v42; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v43; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v44[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v45; // [rsp+50h] [rbp-90h]
  unsigned __int8 *v46; // [rsp+60h] [rbp-80h] BYREF
  __int64 v47; // [rsp+68h] [rbp-78h]
  __int64 *v48; // [rsp+70h] [rbp-70h]
  __int64 v49; // [rsp+78h] [rbp-68h]
  __int64 v50; // [rsp+80h] [rbp-60h]
  int v51; // [rsp+88h] [rbp-58h]
  __int64 v52; // [rsp+90h] [rbp-50h]
  __int64 v53; // [rsp+98h] [rbp-48h]

  v9 = sub_16498A0(a5);
  v10 = *(unsigned __int8 **)(a5 + 48);
  v46 = 0;
  v49 = v9;
  v11 = *(_QWORD *)(a5 + 40);
  v50 = 0;
  v47 = v11;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v48 = (__int64 *)(a5 + 24);
  v44[0] = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)v44, (__int64)v10, 2);
    if ( v46 )
      sub_161E7C0((__int64)&v46, (__int64)v46);
    v46 = v44[0];
    if ( v44[0] )
      sub_1623210((__int64)v44, v44[0], (__int64)&v46);
  }
  v12 = *(_QWORD *)(a1 + 16);
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 )
  {
    v14 = *(_QWORD *)(a5 + 40);
    v15 = *(_QWORD *)(v12 + 8);
    v16 = v13 - 1;
    v17 = (v13 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v18 = (__int64 *)(v15 + 16LL * v17);
    v19 = *v18;
    if ( v14 == *v18 )
    {
LABEL_8:
      v20 = v18[1];
      if ( v20 )
      {
        while ( 1 )
        {
          if ( !sub_13FC520(v20)
            || *(_BYTE *)(a2 + 16) > 0x17u
            && !sub_15CC890(*(_QWORD *)(a1 + 40), *(_QWORD *)(a2 + 40), **(_QWORD **)(v20 + 32)) )
          {
            goto LABEL_23;
          }
          v22 = sub_13FC520(v20);
          v23 = sub_157EBA0(v22);
          v24 = *(unsigned __int8 **)(v23 + 48);
          v47 = *(_QWORD *)(v23 + 40);
          v48 = (__int64 *)(v23 + 24);
          v44[0] = v24;
          if ( v24 )
            break;
          v21 = v46;
          if ( v46 )
            goto LABEL_11;
LABEL_14:
          v20 = *(_QWORD *)v20;
          if ( !v20 )
            goto LABEL_23;
        }
        sub_1623A60((__int64)v44, (__int64)v24, 2);
        v21 = v46;
        if ( v46 )
LABEL_11:
          sub_161E7C0((__int64)&v46, (__int64)v21);
        v46 = v44[0];
        if ( v44[0] )
          sub_1623210((__int64)v44, v44[0], (__int64)&v46);
        goto LABEL_14;
      }
    }
    else
    {
      v25 = 1;
      while ( v19 != -8 )
      {
        v39 = v25 + 1;
        v17 = v16 & (v25 + v17);
        v18 = (__int64 *)(v15 + 16LL * v17);
        v19 = *v18;
        if ( v14 == *v18 )
          goto LABEL_8;
        v25 = v39;
      }
    }
  }
LABEL_23:
  v26 = *(__int64 ***)a2;
  v43 = 257;
  if ( a4 )
  {
    if ( a3 != v26 )
    {
      if ( *(_BYTE *)(a2 + 16) <= 0x10u )
      {
        v27 = sub_15A46C0(38, (__int64 ***)a2, a3, 0);
        v28 = v46;
        a2 = v27;
        goto LABEL_30;
      }
      v32 = a2;
      v33 = 38;
      v45 = 257;
      v31 = (__int64)a3;
      goto LABEL_34;
    }
LABEL_41:
    v28 = v46;
LABEL_30:
    if ( v28 )
      sub_161E7C0((__int64)&v46, (__int64)v28);
    return a2;
  }
  if ( a3 == v26 )
    goto LABEL_41;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
    v29 = sub_15A46C0(37, (__int64 ***)a2, a3, 0);
    v28 = v46;
    a2 = v29;
    goto LABEL_30;
  }
  v31 = (__int64)a3;
  v32 = a2;
  v45 = 257;
  v33 = 37;
LABEL_34:
  a2 = sub_15FDBD0(v33, v32, v31, (__int64)v44, 0);
  if ( v47 )
  {
    v34 = v48;
    sub_157E9D0(v47 + 40, a2);
    v35 = *(_QWORD *)(a2 + 24);
    v36 = *v34;
    *(_QWORD *)(a2 + 32) = v34;
    v36 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 24) = v36 | v35 & 7;
    *(_QWORD *)(v36 + 8) = a2 + 24;
    *v34 = *v34 & 7 | (a2 + 24);
  }
  sub_164B780(a2, &v42);
  if ( v46 )
  {
    v41 = v46;
    sub_1623A60((__int64)&v41, (__int64)v46, 2);
    v37 = *(_QWORD *)(a2 + 48);
    if ( v37 )
      sub_161E7C0(a2 + 48, v37);
    v38 = v41;
    *(_QWORD *)(a2 + 48) = v41;
    if ( v38 )
      sub_1623210((__int64)&v41, v38, a2 + 48);
    goto LABEL_41;
  }
  return a2;
}
