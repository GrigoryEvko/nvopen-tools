// Function: sub_1D634F0
// Address: 0x1d634f0
//
__int64 __fastcall sub_1D634F0(__int64 a1, __int64 a2, char a3)
{
  __int64 ***v6; // rax
  __int64 *v7; // r14
  __int64 **v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rax
  __int64 **v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 *v16; // r14
  unsigned int v17; // ebx
  _QWORD *v18; // rax
  _QWORD *v19; // r12
  __int64 v20; // rdi
  unsigned __int64 *v21; // r13
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rsi
  __int64 v42; // rdx
  unsigned __int8 *v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rsi
  __int64 v50; // rsi
  __int64 v51; // rdx
  unsigned __int8 *v52; // rsi
  __int64 *v53; // [rsp+0h] [rbp-90h]
  __int64 v54; // [rsp+0h] [rbp-90h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  _BYTE *v56; // [rsp+8h] [rbp-88h]
  _BYTE *v57; // [rsp+8h] [rbp-88h]
  __int64 *v58; // [rsp+8h] [rbp-88h]
  __int64 v59; // [rsp+8h] [rbp-88h]
  __int64 v60; // [rsp+8h] [rbp-88h]
  __int64 v61; // [rsp+8h] [rbp-88h]
  __int64 v62; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v63; // [rsp+18h] [rbp-78h] BYREF
  __int64 v64[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v65; // [rsp+30h] [rbp-60h]
  __int64 v66[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v67; // [rsp+50h] [rbp-40h]

  v6 = *(__int64 ****)(a1 + 8);
  v7 = *(__int64 **)a1;
  v65 = 257;
  v8 = *v6;
  if ( *v6 != *(__int64 ***)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v67 = 257;
      v36 = sub_15FDE70((_QWORD *)a2, (__int64)v8, (__int64)v66, 0);
      v37 = v7[1];
      a2 = v36;
      if ( v37 )
      {
        v58 = (__int64 *)v7[2];
        sub_157E9D0(v37 + 40, v36);
        v38 = *v58;
        v39 = *(_QWORD *)(a2 + 24) & 7LL;
        *(_QWORD *)(a2 + 32) = v58;
        v38 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a2 + 24) = v38 | v39;
        *(_QWORD *)(v38 + 8) = a2 + 24;
        *v58 = *v58 & 7 | (a2 + 24);
      }
      sub_164B780(a2, v64);
      v40 = *v7;
      if ( *v7 )
      {
        v63 = (unsigned __int8 *)*v7;
        sub_1623A60((__int64)&v63, v40, 2);
        v41 = *(_QWORD *)(a2 + 48);
        v42 = a2 + 48;
        if ( v41 )
        {
          sub_161E7C0(a2 + 48, v41);
          v42 = a2 + 48;
        }
        v43 = v63;
        *(_QWORD *)(a2 + 48) = v63;
        if ( v43 )
          sub_1623210((__int64)&v63, v43, v42);
      }
      v7 = *(__int64 **)a1;
      v6 = *(__int64 ****)(a1 + 8);
    }
    else
    {
      v9 = sub_15A45D0((__int64 ***)a2, v8);
      v7 = *(__int64 **)a1;
      a2 = v9;
      v6 = *(__int64 ****)(a1 + 8);
    }
  }
  v65 = 257;
  v10 = (__int64 *)*v6;
  v11 = **(_QWORD **)(*(_QWORD *)(a1 + 16) - 24LL);
  if ( *(_BYTE *)(v11 + 8) == 16 )
    v11 = **(_QWORD **)(v11 + 16);
  v12 = (__int64 **)sub_1647190(v10, *(_DWORD *)(v11 + 8) >> 8);
  v13 = *(_QWORD *)(a1 + 16);
  v14 = *(_QWORD *)(v13 - 24);
  if ( v12 != *(__int64 ***)v14 )
  {
    if ( *(_BYTE *)(v14 + 16) > 0x10u )
    {
      v67 = 257;
      v44 = sub_15FDBD0(47, v14, (__int64)v12, (__int64)v66, 0);
      v45 = v7[1];
      v46 = v44;
      if ( v45 )
      {
        v59 = v44;
        v53 = (__int64 *)v7[2];
        sub_157E9D0(v45 + 40, v44);
        v46 = v59;
        v47 = *(_QWORD *)(v59 + 24);
        v48 = *v53;
        *(_QWORD *)(v59 + 32) = v53;
        v48 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v59 + 24) = v48 | v47 & 7;
        *(_QWORD *)(v48 + 8) = v59 + 24;
        *v53 = *v53 & 7 | (v59 + 24);
      }
      v60 = v46;
      sub_164B780(v46, v64);
      v49 = *v7;
      v14 = v60;
      if ( *v7 )
      {
        v63 = (unsigned __int8 *)*v7;
        sub_1623A60((__int64)&v63, v49, 2);
        v14 = v60;
        v50 = *(_QWORD *)(v60 + 48);
        v51 = v60 + 48;
        if ( v50 )
        {
          v54 = v60;
          v61 = v60 + 48;
          sub_161E7C0(v61, v50);
          v14 = v54;
          v51 = v61;
        }
        v52 = v63;
        *(_QWORD *)(v14 + 48) = v63;
        if ( v52 )
        {
          v62 = v14;
          sub_1623210((__int64)&v63, v52, v51);
          v14 = v62;
        }
      }
      v13 = *(_QWORD *)(a1 + 16);
    }
    else
    {
      v15 = sub_15A46C0(47, *(__int64 ****)(v13 - 24), v12, 0);
      v13 = *(_QWORD *)(a1 + 16);
      v14 = v15;
    }
  }
  v16 = *(__int64 **)a1;
  if ( **(_BYTE **)(a1 + 24) )
  {
    if ( a3 )
    {
      v56 = (_BYTE *)v14;
      v67 = 257;
      v28 = (_QWORD *)sub_16498A0(v13);
      v29 = sub_1643350(v28);
      v30 = sub_159C470(v29, 1, 0);
      v31 = sub_12815B0(v16, **(_QWORD **)(a1 + 8), v56, v30, (__int64)v66);
      v16 = *(__int64 **)a1;
      v13 = *(_QWORD *)(a1 + 16);
      v14 = v31;
      goto LABEL_11;
    }
  }
  else
  {
    if ( a3 )
    {
LABEL_11:
      v17 = (unsigned int)(1 << (*(unsigned __int16 *)(v13 + 18) >> 1) >> 1) >> 1;
      goto LABEL_12;
    }
    v57 = (_BYTE *)v14;
    v67 = 257;
    v32 = (_QWORD *)sub_16498A0(v13);
    v33 = sub_1643350(v32);
    v34 = sub_159C470(v33, 1, 0);
    v35 = sub_12815B0(v16, **(_QWORD **)(a1 + 8), v57, v34, (__int64)v66);
    v16 = *(__int64 **)a1;
    v13 = *(_QWORD *)(a1 + 16);
    v14 = v35;
  }
  v17 = 1 << (*(unsigned __int16 *)(v13 + 18) >> 1) >> 1;
LABEL_12:
  v55 = v14;
  v67 = 257;
  v18 = sub_1648A60(64, 2u);
  v19 = v18;
  if ( v18 )
    sub_15F9650((__int64)v18, a2, v55, 0, 0);
  v20 = v16[1];
  if ( v20 )
  {
    v21 = (unsigned __int64 *)v16[2];
    sub_157E9D0(v20 + 40, (__int64)v19);
    v22 = v19[3];
    v23 = *v21;
    v19[4] = v21;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    v19[3] = v23 | v22 & 7;
    *(_QWORD *)(v23 + 8) = v19 + 3;
    *v21 = *v21 & 7 | (unsigned __int64)(v19 + 3);
  }
  sub_164B780((__int64)v19, v66);
  v24 = *v16;
  if ( *v16 )
  {
    v64[0] = *v16;
    sub_1623A60((__int64)v64, v24, 2);
    v25 = v19[6];
    if ( v25 )
      sub_161E7C0((__int64)(v19 + 6), v25);
    v26 = (unsigned __int8 *)v64[0];
    v19[6] = v64[0];
    if ( v26 )
      sub_1623210((__int64)v64, v26, (__int64)(v19 + 6));
  }
  return sub_15F9450((__int64)v19, v17);
}
