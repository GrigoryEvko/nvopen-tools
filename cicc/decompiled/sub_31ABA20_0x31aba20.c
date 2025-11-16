// Function: sub_31ABA20
// Address: 0x31aba20
//
__int64 __fastcall sub_31ABA20(__int64 *a1, __int64 a2, __int64 a3)
{
  int v3; // r14d
  char v4; // r13
  int *v5; // rbx
  __int64 v6; // r14
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 *v26; // rax
  unsigned int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // r8
  __int64 *v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r9
  __int64 v37; // r8
  __int64 *v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r9
  __int64 v43; // r8
  __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // r12
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 *v59; // rax
  unsigned int v60; // eax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r9
  __int64 v64; // r8
  __int64 *v65; // rax
  unsigned int v66; // eax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r9
  __int64 v70; // r8
  __int64 v71; // [rsp+0h] [rbp-D0h]
  __int64 v72; // [rsp+8h] [rbp-C8h]
  __int64 v73; // [rsp+8h] [rbp-C8h]
  __int64 v74; // [rsp+8h] [rbp-C8h]
  __int64 v75; // [rsp+8h] [rbp-C8h]
  __int64 v76; // [rsp+8h] [rbp-C8h]
  __int64 v77; // [rsp+8h] [rbp-C8h]
  __int64 v78; // [rsp+8h] [rbp-C8h]
  __int64 v79; // [rsp+8h] [rbp-C8h]
  bool v80; // [rsp+13h] [rbp-BDh]
  unsigned int v81; // [rsp+14h] [rbp-BCh]
  __int64 v82; // [rsp+18h] [rbp-B8h]
  __int64 v83; // [rsp+20h] [rbp-B0h]
  __int64 v84; // [rsp+20h] [rbp-B0h]
  __int64 v85; // [rsp+20h] [rbp-B0h]
  __int64 v86; // [rsp+20h] [rbp-B0h]
  __int64 v87; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v88; // [rsp+38h] [rbp-98h]
  __int64 v89; // [rsp+40h] [rbp-90h]
  __int64 v90; // [rsp+48h] [rbp-88h] BYREF
  unsigned int v91; // [rsp+50h] [rbp-80h]
  unsigned int v92; // [rsp+88h] [rbp-48h] BYREF
  int v93; // [rsp+8Ch] [rbp-44h]
  __int64 v94; // [rsp+90h] [rbp-40h]
  char v95; // [rsp+98h] [rbp-38h]

  v3 = a3;
  v4 = BYTE4(a3);
  v81 = a3;
  LODWORD(v5) = sub_31A5290((__int64)a1, a2);
  if ( !(_BYTE)v5 && !v4 )
  {
    if ( v3 == 1 )
    {
      LODWORD(v5) = 1;
      return (unsigned int)v5;
    }
    v6 = *(_QWORD *)(a1[2] + 112);
    v80 = sub_D97040(v6, *(_QWORD *)(a2 + 8));
    if ( v80 )
    {
      v8 = sub_DD8400(v6, a2);
      v9 = *a1;
      LOBYTE(v87) = 0;
      v10 = (__int64)v8;
      v83 = v9;
      sub_31A46A0((__int64)v8, &v87);
      if ( (_BYTE)v87 )
      {
        v87 = v6;
        v44 = &v90;
        v88 = 0;
        v89 = 1;
        do
        {
          *v44 = -4096;
          v44 += 2;
        }
        while ( v44 != (__int64 *)&v92 );
        v94 = v83;
        v93 = 0;
        v92 = v81;
        v95 = 0;
        v82 = v10;
        if ( !sub_DADE90(v87, v10, v83) )
          v82 = sub_31AA7F0((__int64)&v87, v10, v45, v46, v47, v48);
        if ( v95 )
          v82 = sub_D970F0(v6);
        if ( (v89 & 1) == 0 )
          sub_C7D6A0(v90, 16LL * v91, 8);
      }
      else
      {
        v82 = sub_D970F0(v6);
      }
      if ( !sub_D96A50(v82) )
      {
        v11 = v81 - 1LL;
        v84 = v11;
        if ( v11 >> 2 > 0 )
        {
          v5 = (int *)&v92;
          v71 = v11 - 4 * (v11 >> 2);
          while ( 1 )
          {
            v19 = *a1;
            LOBYTE(v87) = 0;
            v75 = v19;
            sub_31A46A0(v10, &v87);
            if ( (_BYTE)v87 )
            {
              v87 = v6;
              v20 = &v90;
              v88 = 0;
              v89 = 1;
              do
              {
                *v20 = -4096;
                v20 += 2;
              }
              while ( v20 != (__int64 *)&v92 );
              v94 = v75;
              v95 = 0;
              v92 = v81;
              v93 = v84;
              LOBYTE(v21) = sub_DADE90(v87, v10, v75);
              v25 = v21;
              v12 = v10;
              if ( !(_BYTE)v25 )
                v12 = sub_31AA7F0((__int64)&v87, v10, v22, v23, v25, v24);
              if ( v95 )
                v12 = sub_D970F0(v6);
              if ( (v89 & 1) == 0 )
              {
                v76 = v12;
                sub_C7D6A0(v90, 16LL * v91, 8);
                v12 = v76;
              }
            }
            else
            {
              v12 = sub_D970F0(v6);
            }
            if ( v82 != v12 )
            {
              LOBYTE(v5) = v84 == 0;
              return (unsigned int)v5;
            }
            v13 = *a1;
            LOBYTE(v87) = 0;
            v72 = v13;
            sub_31A46A0(v10, &v87);
            if ( (_BYTE)v87 )
            {
              v87 = v6;
              v26 = &v90;
              v88 = 0;
              v89 = 1;
              do
              {
                *v26 = -4096;
                v26 += 2;
              }
              while ( v26 != (__int64 *)&v92 );
              v94 = v72;
              v95 = 0;
              v92 = v81;
              v93 = v84 - 1;
              LOBYTE(v27) = sub_DADE90(v87, v10, v72);
              v31 = v27;
              v14 = v10;
              if ( !(_BYTE)v31 )
                v14 = sub_31AA7F0((__int64)&v87, v10, v28, v29, v31, v30);
              if ( v95 )
                v14 = sub_D970F0(v6);
              if ( (v89 & 1) == 0 )
              {
                v77 = v14;
                sub_C7D6A0(v90, 16LL * v91, 8);
                v14 = v77;
              }
            }
            else
            {
              v14 = sub_D970F0(v6);
            }
            if ( v82 != v14 )
            {
              LOBYTE(v5) = v84 == 1;
              return (unsigned int)v5;
            }
            v15 = *a1;
            LOBYTE(v87) = 0;
            v73 = v15;
            sub_31A46A0(v10, &v87);
            if ( (_BYTE)v87 )
            {
              v87 = v6;
              v32 = &v90;
              v88 = 0;
              v89 = 1;
              do
              {
                *v32 = -4096;
                v32 += 2;
              }
              while ( v32 != (__int64 *)&v92 );
              v94 = v73;
              v95 = 0;
              v92 = v81;
              v93 = v84 - 2;
              LOBYTE(v33) = sub_DADE90(v87, v10, v73);
              v37 = v33;
              v16 = v10;
              if ( !(_BYTE)v37 )
                v16 = sub_31AA7F0((__int64)&v87, v10, v34, v35, v37, v36);
              if ( v95 )
                v16 = sub_D970F0(v6);
              if ( (v89 & 1) == 0 )
              {
                v78 = v16;
                sub_C7D6A0(v90, 16LL * v91, 8);
                v16 = v78;
              }
            }
            else
            {
              v16 = sub_D970F0(v6);
            }
            if ( v82 != v16 )
            {
              LOBYTE(v5) = v84 == 2;
              return (unsigned int)v5;
            }
            v17 = *a1;
            LOBYTE(v87) = 0;
            v74 = v17;
            sub_31A46A0(v10, &v87);
            if ( (_BYTE)v87 )
            {
              v87 = v6;
              v38 = &v90;
              v88 = 0;
              v89 = 1;
              do
              {
                *v38 = -4096;
                v38 += 2;
              }
              while ( v38 != (__int64 *)&v92 );
              v94 = v74;
              v95 = 0;
              v92 = v81;
              v93 = v84 - 3;
              LOBYTE(v39) = sub_DADE90(v87, v10, v74);
              v43 = v39;
              v18 = v10;
              if ( !(_BYTE)v43 )
                v18 = sub_31AA7F0((__int64)&v87, v10, v40, v41, v43, v42);
              if ( v95 )
                v18 = sub_D970F0(v6);
              if ( (v89 & 1) == 0 )
              {
                v79 = v18;
                sub_C7D6A0(v90, 16LL * v91, 8);
                v18 = v79;
              }
            }
            else
            {
              v18 = sub_D970F0(v6);
            }
            if ( v82 != v18 )
              break;
            v84 -= 4;
            if ( v84 == v71 )
              goto LABEL_65;
          }
          LOBYTE(v5) = v84 == 3;
          return (unsigned int)v5;
        }
LABEL_65:
        if ( v84 != 2 )
        {
          if ( v84 != 3 )
          {
            if ( v84 != 1 )
              goto LABEL_68;
            goto LABEL_79;
          }
          LOBYTE(v87) = 0;
          v49 = *a1;
          sub_31A46A0(v10, &v87);
          if ( (_BYTE)v87 )
          {
            v87 = v6;
            v65 = &v90;
            v88 = 0;
            v89 = 1;
            do
            {
              *v65 = -4096;
              v65 += 2;
            }
            while ( v65 != (__int64 *)&v92 );
            v94 = v49;
            v93 = 3;
            v92 = v81;
            v95 = 0;
            LOBYTE(v66) = sub_DADE90(v87, v10, v49);
            v70 = v66;
            v50 = v10;
            if ( !(_BYTE)v70 )
              v50 = sub_31AA7F0((__int64)&v87, v10, v67, v68, v70, v69);
            if ( v95 )
              v50 = sub_D970F0(v6);
            if ( (v89 & 1) == 0 )
            {
              v86 = v50;
              sub_C7D6A0(v90, 16LL * v91, 8);
              v50 = v86;
            }
          }
          else
          {
            v50 = sub_D970F0(v6);
          }
          LODWORD(v5) = 0;
          if ( v82 != v50 )
            return (unsigned int)v5;
        }
        LOBYTE(v87) = 0;
        v51 = *a1;
        sub_31A46A0(v10, &v87);
        if ( (_BYTE)v87 )
        {
          v87 = v6;
          v59 = &v90;
          v88 = 0;
          v89 = 1;
          do
          {
            *v59 = -4096;
            v59 += 2;
          }
          while ( v59 != (__int64 *)&v92 );
          v94 = v51;
          v93 = 2;
          v92 = v81;
          v95 = 0;
          LOBYTE(v60) = sub_DADE90(v87, v10, v51);
          v64 = v60;
          v52 = v10;
          if ( !(_BYTE)v64 )
            v52 = sub_31AA7F0((__int64)&v87, v10, v61, v62, v64, v63);
          if ( v95 )
            v52 = sub_D970F0(v6);
          if ( (v89 & 1) == 0 )
          {
            v85 = v52;
            sub_C7D6A0(v90, 16LL * v91, 8);
            v52 = v85;
          }
        }
        else
        {
          v52 = sub_D970F0(v6);
        }
        LODWORD(v5) = 0;
        if ( v82 != v52 )
          return (unsigned int)v5;
LABEL_79:
        LOBYTE(v87) = 0;
        v53 = *a1;
        sub_31A46A0(v10, &v87);
        if ( (_BYTE)v87 )
        {
          v87 = v6;
          v54 = &v90;
          v88 = 0;
          v89 = 1;
          do
          {
            *v54 = -4096;
            v54 += 2;
          }
          while ( v54 != (__int64 *)&v92 );
          v94 = v53;
          v93 = 1;
          v92 = v81;
          v95 = 0;
          if ( !sub_DADE90(v87, v10, v53) )
            v10 = sub_31AA7F0((__int64)&v87, v10, v55, v56, v57, v58);
          if ( v95 )
            v10 = sub_D970F0(v6);
          if ( (v89 & 1) == 0 )
            sub_C7D6A0(v90, 16LL * v91, 8);
        }
        else
        {
          v10 = sub_D970F0(v6);
        }
        LODWORD(v5) = 0;
        if ( v82 != v10 )
          return (unsigned int)v5;
LABEL_68:
        LODWORD(v5) = v80;
      }
    }
  }
  return (unsigned int)v5;
}
