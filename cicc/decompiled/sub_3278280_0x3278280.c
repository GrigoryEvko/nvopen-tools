// Function: sub_3278280
// Address: 0x3278280
//
__int64 __fastcall sub_3278280(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  unsigned __int16 *v9; // rdx
  unsigned __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int16 *v15; // rdx
  unsigned __int16 v16; // ax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  char v20; // al
  unsigned __int16 *v21; // rdx
  unsigned __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  char v26; // al
  unsigned __int16 *v27; // rdx
  unsigned __int16 v28; // ax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int16 *v36; // rax
  unsigned __int16 v37; // dx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  char v41; // al
  unsigned __int16 *v42; // rax
  unsigned __int16 v43; // dx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  char v47; // al
  unsigned __int16 *v48; // rdx
  unsigned __int16 v49; // ax
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rdx
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned __int16 v58; // [rsp+20h] [rbp-100h] BYREF
  __int64 v59; // [rsp+28h] [rbp-F8h]
  __int64 v60; // [rsp+30h] [rbp-F0h]
  __int64 v61; // [rsp+38h] [rbp-E8h]
  unsigned __int16 v62; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-D8h]
  __int64 v64; // [rsp+50h] [rbp-D0h]
  __int64 v65; // [rsp+58h] [rbp-C8h]
  unsigned __int16 v66; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+68h] [rbp-B8h]
  __int64 v68; // [rsp+70h] [rbp-B0h]
  __int64 v69; // [rsp+78h] [rbp-A8h]
  unsigned __int16 v70; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v71; // [rsp+88h] [rbp-98h]
  __int64 v72; // [rsp+90h] [rbp-90h]
  __int64 v73; // [rsp+98h] [rbp-88h]
  unsigned __int16 v74; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v75; // [rsp+A8h] [rbp-78h]
  __int64 v76; // [rsp+B0h] [rbp-70h]
  __int64 v77; // [rsp+B8h] [rbp-68h]
  __int64 v78; // [rsp+C0h] [rbp-60h]
  __int64 v79; // [rsp+C8h] [rbp-58h]
  __int64 v80; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v81; // [rsp+D8h] [rbp-48h]
  __int64 v82; // [rsp+E0h] [rbp-40h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-38h]

  v4 = (a2 - a1) >> 6;
  v5 = (a2 - a1) >> 4;
  if ( v4 <= 0 )
  {
    v6 = a1;
LABEL_27:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          return a2;
LABEL_55:
        if ( *(_DWORD *)(v6 + 12) != a3 )
          return v6;
        v48 = *(unsigned __int16 **)(*(_QWORD *)v6 + 48LL);
        v49 = *v48;
        v50 = *((_QWORD *)v48 + 1);
        LOWORD(v80) = v49;
        v81 = v50;
        if ( !v49 )
        {
          v82 = sub_3007260((__int64)&v80);
          v83 = v51;
          v52 = v82;
          v53 = v83;
LABEL_58:
          v80 = v52;
          LOBYTE(v81) = v53;
          if ( sub_CA1930(&v80) == a3 && !(*(_DWORD *)(v6 + 8) % a3) )
            return a2;
          return v6;
        }
        if ( v49 != 1 && (unsigned __int16)(v49 - 504) > 7u )
        {
          v54 = 16LL * (v49 - 1);
          v52 = *(_QWORD *)&byte_444C4A0[v54];
          v53 = byte_444C4A0[v54 + 8];
          goto LABEL_58;
        }
LABEL_72:
        BUG();
      }
      if ( *(_DWORD *)(v6 + 12) != a3 )
        return v6;
      v36 = *(unsigned __int16 **)(*(_QWORD *)v6 + 48LL);
      v37 = *v36;
      v38 = *((_QWORD *)v36 + 1);
      v74 = v37;
      v75 = v38;
      if ( v37 )
      {
        if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
          goto LABEL_72;
        v56 = 16LL * (v37 - 1);
        v40 = *(_QWORD *)&byte_444C4A0[v56];
        v41 = byte_444C4A0[v56 + 8];
      }
      else
      {
        v76 = sub_3007260((__int64)&v74);
        v77 = v39;
        v40 = v76;
        v41 = v77;
      }
      v82 = v40;
      LOBYTE(v83) = v41;
      if ( sub_CA1930(&v82) != a3 || *(_DWORD *)(v6 + 8) % a3 )
        return v6;
      v6 += 16;
    }
    if ( *(_DWORD *)(v6 + 12) != a3 )
      return v6;
    v42 = *(unsigned __int16 **)(*(_QWORD *)v6 + 48LL);
    v43 = *v42;
    v44 = *((_QWORD *)v42 + 1);
    LOWORD(v82) = v43;
    v83 = v44;
    if ( v43 )
    {
      if ( v43 == 1 || (unsigned __int16)(v43 - 504) <= 7u )
        goto LABEL_72;
      v55 = 16LL * (v43 - 1);
      v46 = *(_QWORD *)&byte_444C4A0[v55];
      v47 = byte_444C4A0[v55 + 8];
    }
    else
    {
      v78 = sub_3007260((__int64)&v82);
      v79 = v45;
      v46 = v78;
      v47 = v79;
    }
    v82 = v46;
    LOBYTE(v83) = v47;
    if ( sub_CA1930(&v82) != a3 || *(_DWORD *)(v6 + 8) % a3 )
      return v6;
    v6 += 16;
    goto LABEL_55;
  }
  v6 = a1;
  v7 = a1 + (v4 << 6);
  while ( 1 )
  {
    if ( *(_DWORD *)(v6 + 12) != a3 )
      return v6;
    v9 = *(unsigned __int16 **)(*(_QWORD *)v6 + 48LL);
    v10 = *v9;
    v11 = *((_QWORD *)v9 + 1);
    v58 = v10;
    v59 = v11;
    if ( v10 )
    {
      if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
        goto LABEL_72;
      v13 = 16LL * (v10 - 1);
      v12 = *(_QWORD *)&byte_444C4A0[v13];
      LOBYTE(v13) = byte_444C4A0[v13 + 8];
    }
    else
    {
      v12 = sub_3007260((__int64)&v58);
      v60 = v12;
      v61 = v13;
    }
    v82 = v12;
    LOBYTE(v83) = v13;
    if ( sub_CA1930(&v82) != a3 || *(_DWORD *)(v6 + 8) % a3 )
      return v6;
    v14 = v6 + 16;
    if ( *(_DWORD *)(v6 + 28) != a3 )
      return v14;
    v15 = *(unsigned __int16 **)(*(_QWORD *)(v6 + 16) + 48LL);
    v16 = *v15;
    v17 = *((_QWORD *)v15 + 1);
    v62 = v16;
    v63 = v17;
    if ( v16 )
    {
      if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
        goto LABEL_72;
      v33 = 16LL * (v16 - 1);
      v19 = *(_QWORD *)&byte_444C4A0[v33];
      v20 = byte_444C4A0[v33 + 8];
    }
    else
    {
      v14 = v6 + 16;
      v64 = sub_3007260((__int64)&v62);
      v65 = v18;
      v19 = v64;
      v20 = v65;
    }
    v82 = v19;
    LOBYTE(v83) = v20;
    if ( a3 != sub_CA1930(&v82) )
      return v14;
    if ( *(_DWORD *)(v6 + 24) % a3 )
      return v14;
    v14 = v6 + 32;
    if ( *(_DWORD *)(v6 + 44) != a3 )
      return v14;
    v21 = *(unsigned __int16 **)(*(_QWORD *)(v6 + 32) + 48LL);
    v22 = *v21;
    v23 = *((_QWORD *)v21 + 1);
    v66 = v22;
    v67 = v23;
    if ( v22 )
    {
      if ( v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
        goto LABEL_72;
      v34 = 16LL * (v22 - 1);
      v25 = *(_QWORD *)&byte_444C4A0[v34];
      v26 = byte_444C4A0[v34 + 8];
    }
    else
    {
      v14 = v6 + 32;
      v68 = sub_3007260((__int64)&v66);
      v69 = v24;
      v25 = v68;
      v26 = v69;
    }
    v82 = v25;
    LOBYTE(v83) = v26;
    if ( a3 != sub_CA1930(&v82) )
      return v14;
    if ( *(_DWORD *)(v6 + 40) % a3 )
      return v14;
    v14 = v6 + 48;
    if ( *(_DWORD *)(v6 + 60) != a3 )
      return v14;
    v27 = *(unsigned __int16 **)(*(_QWORD *)(v6 + 48) + 48LL);
    v28 = *v27;
    v29 = *((_QWORD *)v27 + 1);
    v70 = v28;
    v71 = v29;
    if ( v28 )
    {
      if ( v28 == 1 || (unsigned __int16)(v28 - 504) <= 7u )
        goto LABEL_72;
      v35 = 16LL * (v28 - 1);
      v31 = *(_QWORD *)&byte_444C4A0[v35];
      v32 = byte_444C4A0[v35 + 8];
    }
    else
    {
      v14 = v6 + 48;
      v72 = sub_3007260((__int64)&v70);
      v73 = v30;
      v31 = v72;
      v32 = v73;
    }
    v82 = v31;
    LOBYTE(v83) = v32;
    if ( a3 != sub_CA1930(&v82) || *(_DWORD *)(v6 + 56) % a3 )
      return v14;
    v6 += 64;
    if ( v7 == v6 )
    {
      v5 = (a2 - v6) >> 4;
      goto LABEL_27;
    }
  }
}
