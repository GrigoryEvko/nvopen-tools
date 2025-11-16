// Function: sub_128B750
// Address: 0x128b750
//
__int64 __fastcall sub_128B750(__int64 a1, _BYTE *a2, __int64 *a3, __int64 a4, char a5)
{
  __int64 v8; // r15
  __int64 *v9; // r10
  __int64 *v10; // r9
  __int64 *v11; // rcx
  unsigned int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // r15
  __int64 v16; // rax
  _BYTE *v17; // r12
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rbx
  _QWORD *v22; // r12
  __int64 v23; // rdx
  __int64 *v25; // rdi
  unsigned int v26; // r14d
  unsigned int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // r10
  __int64 v32; // rcx
  __int64 v33; // rdi
  __int64 *v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 *v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdi
  unsigned __int64 *v48; // r13
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 *v54; // r10
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 *v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 *v61; // rdx
  __int64 v62; // rsi
  __int64 *v63; // [rsp+0h] [rbp-A0h]
  __int64 *v64; // [rsp+0h] [rbp-A0h]
  __int64 *v65; // [rsp+8h] [rbp-98h]
  __int64 *v66; // [rsp+8h] [rbp-98h]
  __int64 *v67; // [rsp+8h] [rbp-98h]
  __int64 *v68; // [rsp+8h] [rbp-98h]
  __int64 *v69; // [rsp+10h] [rbp-90h]
  __int64 v70; // [rsp+10h] [rbp-90h]
  __int64 *v71; // [rsp+10h] [rbp-90h]
  __int64 *v72; // [rsp+10h] [rbp-90h]
  __int64 *v73; // [rsp+10h] [rbp-90h]
  __int64 *v74; // [rsp+10h] [rbp-90h]
  __int64 v75; // [rsp+10h] [rbp-90h]
  __int64 *v76; // [rsp+10h] [rbp-90h]
  __int64 *v77; // [rsp+10h] [rbp-90h]
  __int64 *v78; // [rsp+18h] [rbp-88h] BYREF
  __int64 v79; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v80[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v81; // [rsp+40h] [rbp-60h]
  _QWORD v82[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v83; // [rsp+60h] [rbp-40h]

  v78 = a3;
  v8 = *(_QWORD *)a2;
  if ( (unsigned __int8)sub_127B3A0(a4) || (v26 = *(_DWORD *)(*v78 + 8) >> 8, v27 = sub_127B390(), v27 <= v26) )
  {
    v9 = *(__int64 **)(a1 + 8);
    v10 = v78;
  }
  else
  {
    v28 = sub_1644900(*(_QWORD *)(a1 + 16), v27);
    v10 = v78;
    v81 = 259;
    v9 = *(__int64 **)(a1 + 8);
    v80[0] = "idx.ext";
    if ( v28 != *v78 )
    {
      if ( *((_BYTE *)v78 + 16) > 0x10u )
      {
        v83 = 257;
        v74 = v9;
        v53 = sub_15FDBD0(37, v78, v28, v82, 0);
        v54 = v74;
        v55 = v53;
        v56 = v74[1];
        if ( v56 )
        {
          v57 = (__int64 *)v74[2];
          v64 = v74;
          v75 = v55;
          v67 = v57;
          sub_157E9D0(v56 + 40, v55);
          v55 = v75;
          v54 = v64;
          v58 = *v67;
          v59 = *(_QWORD *)(v75 + 24);
          *(_QWORD *)(v75 + 32) = v67;
          v58 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v75 + 24) = v58 | v59 & 7;
          *(_QWORD *)(v58 + 8) = v75 + 24;
          *v67 = *v67 & 7 | (v75 + 24);
        }
        v68 = v54;
        v76 = (__int64 *)v55;
        sub_164B780(v55, v80);
        v10 = v76;
        v60 = *v68;
        if ( *v68 )
        {
          v79 = *v68;
          sub_1623A60(&v79, v60, 2);
          v10 = v76;
          v61 = v76 + 6;
          if ( v76[6] )
          {
            sub_161E7C0(v76 + 6);
            v10 = v76;
            v61 = v76 + 6;
          }
          v62 = v79;
          v10[6] = v79;
          if ( v62 )
          {
            v77 = v10;
            sub_1623210(&v79, v62, v61);
            v10 = v77;
          }
        }
        v9 = *(__int64 **)(a1 + 8);
      }
      else
      {
        v29 = sub_15A46C0(37, v78, v28, 0);
        v9 = *(__int64 **)(a1 + 8);
        v10 = (__int64 *)v29;
      }
    }
    v78 = v10;
  }
  v80[0] = "sub.ptr.neg";
  v81 = 259;
  if ( *((_BYTE *)v10 + 16) > 0x10u )
  {
    v69 = v9;
    v83 = 257;
    v30 = sub_15FB530(v10, v82, 0);
    v31 = v69;
    v32 = v30;
    v33 = v69[1];
    if ( v33 )
    {
      v34 = (__int64 *)v69[2];
      v63 = v69;
      v70 = v30;
      v65 = v34;
      sub_157E9D0(v33 + 40, v30);
      v32 = v70;
      v31 = v63;
      v35 = *v65;
      v36 = *(_QWORD *)(v70 + 24);
      *(_QWORD *)(v70 + 32) = v65;
      v35 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v70 + 24) = v35 | v36 & 7;
      *(_QWORD *)(v35 + 8) = v70 + 24;
      *v65 = *v65 & 7 | (v70 + 24);
    }
    v66 = v31;
    v71 = (__int64 *)v32;
    sub_164B780(v32, v80);
    v11 = v71;
    v37 = *v66;
    if ( *v66 )
    {
      v79 = *v66;
      sub_1623A60(&v79, v37, 2);
      v11 = v71;
      v38 = v71 + 6;
      if ( v71[6] )
      {
        sub_161E7C0(v71 + 6);
        v11 = v71;
        v38 = v71 + 6;
      }
      v39 = v79;
      v11[6] = v79;
      if ( v39 )
      {
        v72 = v11;
        sub_1623210(&v79, v39, v38);
        v11 = v72;
      }
    }
  }
  else
  {
    v11 = (__int64 *)sub_15A2B90(v10, 0, 0);
  }
  v78 = v11;
  if ( a5 || *(_BYTE *)(*(_QWORD *)(v8 + 24) + 8LL) == 12 )
  {
    v12 = *(_DWORD *)(v8 + 8);
    v13 = sub_1643330(*(_QWORD *)(a1 + 16));
    v14 = sub_1646BA0(v13, v12 >> 8);
    v15 = *(__int64 **)(a1 + 8);
    v81 = 257;
    if ( v14 == *(_QWORD *)a2 )
    {
      v17 = a2;
    }
    else if ( a2[16] > 0x10u )
    {
      v83 = 257;
      v40 = sub_15FDBD0(47, a2, v14, v82, 0);
      v41 = v15[1];
      v17 = (_BYTE *)v40;
      if ( v41 )
      {
        v73 = (__int64 *)v15[2];
        sub_157E9D0(v41 + 40, v40);
        v42 = *v73;
        v43 = *((_QWORD *)v17 + 3) & 7LL;
        *((_QWORD *)v17 + 4) = v73;
        v42 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v17 + 3) = v42 | v43;
        *(_QWORD *)(v42 + 8) = v17 + 24;
        *v73 = *v73 & 7 | (unsigned __int64)(v17 + 24);
      }
      sub_164B780(v17, v80);
      v44 = *v15;
      if ( *v15 )
      {
        v79 = *v15;
        sub_1623A60(&v79, v44, 2);
        if ( *((_QWORD *)v17 + 6) )
          sub_161E7C0(v17 + 48);
        v45 = v79;
        *((_QWORD *)v17 + 6) = v79;
        if ( v45 )
          sub_1623210(&v79, v45, v17 + 48);
      }
      v15 = *(__int64 **)(a1 + 8);
    }
    else
    {
      v16 = sub_15A46C0(47, a2, v14, 0);
      v15 = *(__int64 **)(a1 + 8);
      v17 = (_BYTE *)v16;
    }
    v18 = *(_QWORD *)(a1 + 16);
    v82[0] = "sub.ptr";
    v83 = 259;
    v19 = sub_1643330(v18);
    v20 = sub_12815B0(v15, v19, v17, (__int64)v78, (__int64)v82);
    v21 = *(__int64 **)(a1 + 8);
    v81 = 257;
    v22 = (_QWORD *)v20;
    v23 = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 != *(_QWORD *)v20 )
    {
      if ( *(_BYTE *)(v20 + 16) > 0x10u )
      {
        v83 = 257;
        v46 = sub_15FDBD0(47, v20, v23, v82, 0);
        v47 = v21[1];
        v22 = (_QWORD *)v46;
        if ( v47 )
        {
          v48 = (unsigned __int64 *)v21[2];
          sub_157E9D0(v47 + 40, v46);
          v49 = v22[3];
          v50 = *v48;
          v22[4] = v48;
          v50 &= 0xFFFFFFFFFFFFFFF8LL;
          v22[3] = v50 | v49 & 7;
          *(_QWORD *)(v50 + 8) = v22 + 3;
          *v48 = *v48 & 7 | (unsigned __int64)(v22 + 3);
        }
        sub_164B780(v22, v80);
        v51 = *v21;
        if ( *v21 )
        {
          v79 = *v21;
          sub_1623A60(&v79, v51, 2);
          if ( v22[6] )
            sub_161E7C0(v22 + 6);
          v52 = v79;
          v22[6] = v79;
          if ( v52 )
            sub_1623210(&v79, v52, v22 + 6);
        }
      }
      else
      {
        return sub_15A46C0(47, v20, v23, 0);
      }
    }
  }
  else
  {
    v25 = *(__int64 **)(a1 + 8);
    v82[0] = "sub.ptr";
    v83 = 259;
    return sub_128B460(v25, 0, a2, &v78, 1, (__int64)v82);
  }
  return (__int64)v22;
}
