// Function: sub_12873B0
// Address: 0x12873b0
//
__int64 __fastcall sub_12873B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r13
  unsigned int v9; // r9d
  __int64 *v10; // rax
  unsigned __int64 v11; // rsi
  unsigned int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rdx
  unsigned int v17; // r9d
  __int64 v18; // rax
  _BOOL4 v19; // edx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  char v24; // al
  __int64 v25; // rdx
  _BOOL4 v26; // edx
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // rdi
  unsigned int v32; // r9d
  __int64 *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rsi
  unsigned int v38; // ecx
  char v39; // al
  unsigned int v40; // eax
  unsigned int v41; // eax
  unsigned int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-B8h]
  unsigned int v46; // [rsp+8h] [rbp-B8h]
  unsigned int v47; // [rsp+8h] [rbp-B8h]
  unsigned int v48; // [rsp+8h] [rbp-B8h]
  unsigned int v49; // [rsp+8h] [rbp-B8h]
  unsigned int v50; // [rsp+8h] [rbp-B8h]
  unsigned int v51; // [rsp+8h] [rbp-B8h]
  __int64 v52; // [rsp+8h] [rbp-B8h]
  unsigned int v53; // [rsp+8h] [rbp-B8h]
  __int64 v54; // [rsp+18h] [rbp-A8h] BYREF
  const char *v55; // [rsp+20h] [rbp-A0h] BYREF
  char v56; // [rsp+30h] [rbp-90h]
  char v57; // [rsp+31h] [rbp-8Fh]
  char v58[16]; // [rsp+40h] [rbp-80h] BYREF
  __int16 v59; // [rsp+50h] [rbp-70h]
  int v60; // [rsp+60h] [rbp-60h] BYREF
  __int64 v61; // [rsp+68h] [rbp-58h]
  unsigned int v62; // [rsp+70h] [rbp-50h]
  __int64 v63; // [rsp+78h] [rbp-48h]
  __int64 v64; // [rsp+80h] [rbp-40h]

  v45 = *(_QWORD *)(a3 + 72);
  sub_1286D80((__int64)&v60, a2, v45, a4, a5);
  v8 = v61;
  v9 = v62;
  v10 = (__int64 *)v45;
  if ( v60 != 1 )
  {
    v11 = *(_QWORD *)(a3 + 8);
    v12 = *(_DWORD *)(*(_QWORD *)v61 + 8LL) >> 8;
    if ( v11 )
    {
      v13 = sub_127A030(a2[4] + 8LL, v11, 0);
      v14 = sub_1646BA0(v13, v12);
      v15 = *(_QWORD *)(a3 + 8);
      v16 = v14;
      if ( *(char *)(v15 + 142) < 0 )
        goto LABEL_5;
    }
    else
    {
      v47 = v62;
      if ( (*(_BYTE *)(a3 + 25) & 1) == 0 && sub_8D2310(*v10) )
      {
        if ( (unsigned int)sub_8D2E30(*(_QWORD *)a3) )
        {
          v43 = sub_8D46C0(*(_QWORD *)a3);
          if ( sub_8D2310(v43) )
          {
            v44 = sub_127A030(a2[4] + 8LL, *(_QWORD *)a3, 0);
            v15 = *(_QWORD *)a3;
            v17 = v47;
            v16 = v44;
            goto LABEL_6;
          }
        }
      }
      v21 = sub_127A030(a2[4] + 8LL, *(_QWORD *)a3, 0);
      v22 = sub_1646BA0(v21, v12);
      v15 = *(_QWORD *)a3;
      v16 = v22;
      if ( *(char *)(*(_QWORD *)a3 + 142LL) < 0 )
        goto LABEL_5;
    }
    if ( *(_BYTE *)(v15 + 140) == 12 )
    {
      v52 = v16;
      v40 = sub_8D4AB0(v15);
      v16 = v52;
      v17 = v40;
      goto LABEL_6;
    }
LABEL_5:
    v17 = *(_DWORD *)(v15 + 136);
LABEL_6:
    v57 = 1;
    v55 = "lvaladjust";
    v56 = 3;
    if ( v16 != *(_QWORD *)v8 )
    {
      v46 = v17;
      if ( *(_BYTE *)(v8 + 16) > 0x10u )
      {
        v59 = 257;
        v30 = sub_15FDBD0(47, v8, v16, v58, 0);
        v31 = a2[7];
        v32 = v46;
        v8 = v30;
        if ( v31 )
        {
          v33 = (__int64 *)a2[8];
          sub_157E9D0(v31 + 40, v30);
          v34 = *(_QWORD *)(v8 + 24);
          v32 = v46;
          v35 = *v33;
          *(_QWORD *)(v8 + 32) = v33;
          v35 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v8 + 24) = v35 | v34 & 7;
          *(_QWORD *)(v35 + 8) = v8 + 24;
          *v33 = *v33 & 7 | (v8 + 24);
        }
        v49 = v32;
        sub_164B780(v8, &v55);
        v36 = a2[6];
        v17 = v49;
        if ( v36 )
        {
          v54 = a2[6];
          sub_1623A60(&v54, v36, 2);
          v17 = v49;
          if ( *(_QWORD *)(v8 + 48) )
          {
            sub_161E7C0(v8 + 48);
            v17 = v49;
          }
          v37 = v54;
          *(_QWORD *)(v8 + 48) = v54;
          if ( v37 )
          {
            v50 = v17;
            sub_1623210(&v54, v37, v8 + 48);
            v17 = v50;
          }
        }
      }
      else
      {
        v18 = sub_15A46C0(47, v8, v16, 0);
        v17 = v46;
        v8 = v18;
      }
    }
    v19 = 0;
    if ( (*(_BYTE *)(v15 + 140) & 0xFB) == 8 )
    {
      v48 = v17;
      v29 = sub_8D4C10(v15, dword_4F077C4 != 2);
      v17 = v48;
      v19 = (v29 & 2) != 0;
    }
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v8;
    *(_DWORD *)(a1 + 40) = v19;
    *(_DWORD *)(a1 + 16) = v17;
    return a1;
  }
  v23 = *(_QWORD *)a3;
  v24 = *(_BYTE *)(*(_QWORD *)a3 + 140LL);
  v25 = *(_QWORD *)a3;
  if ( *(char *)(*(_QWORD *)a3 + 142LL) < 0 )
  {
    v38 = *(_DWORD *)(v23 + 136);
    if ( v62 >= v38 )
      goto LABEL_22;
    goto LABEL_33;
  }
  if ( v24 == 12 )
  {
    v53 = v62;
    v41 = sub_8D4AB0(v23);
    v9 = v53;
    v23 = *(_QWORD *)a3;
    if ( v53 >= v41 )
      goto LABEL_39;
    v25 = *(_QWORD *)a3;
    if ( *(char *)(v23 + 142) >= 0 )
    {
      if ( *(_BYTE *)(v23 + 140) != 12 )
        goto LABEL_21;
      v42 = sub_8D4AB0(v23);
      v23 = *(_QWORD *)a3;
      v9 = v42;
LABEL_39:
      v24 = *(_BYTE *)(v23 + 140);
      goto LABEL_22;
    }
    v38 = *(_DWORD *)(v23 + 136);
LABEL_33:
    v24 = *(_BYTE *)(v25 + 140);
    v9 = v38;
    goto LABEL_22;
  }
  if ( v62 < *(_DWORD *)(v23 + 136) )
  {
LABEL_21:
    v9 = *(_DWORD *)(v25 + 136);
    v24 = *(_BYTE *)(v25 + 140);
    v23 = v25;
  }
LABEL_22:
  v26 = 0;
  if ( (v24 & 0xFB) == 8 )
  {
    v51 = v9;
    v39 = sub_8D4C10(v23, dword_4F077C4 != 2);
    v9 = v51;
    v26 = (v39 & 2) != 0;
  }
  v27 = v63;
  *(_DWORD *)a1 = 1;
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 24) = v27;
  v28 = v64;
  *(_DWORD *)(a1 + 40) = v26;
  *(_QWORD *)(a1 + 32) = v28;
  *(_DWORD *)(a1 + 16) = v9;
  return a1;
}
