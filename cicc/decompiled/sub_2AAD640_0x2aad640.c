// Function: sub_2AAD640
// Address: 0x2aad640
//
__int64 __fastcall sub_2AAD640(__int64 a1)
{
  __int64 v2; // r15
  __int64 v3; // rdi
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r9
  __int64 **v7; // r11
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned __int8 **v15; // rsi
  int v16; // ecx
  unsigned __int8 **v17; // rdx
  unsigned __int64 v18; // rdx
  __int64 v19; // rbx
  int v20; // edx
  int v21; // r12d
  int v22; // eax
  unsigned __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r15
  __int64 v26; // r14
  __int64 v27; // r9
  __int64 **v28; // r11
  __int64 v29; // r8
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v32; // r12
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rbx
  unsigned __int8 **v36; // rsi
  int v37; // ecx
  unsigned __int8 **v38; // rdx
  unsigned __int64 v39; // rdx
  __int64 v40; // rbx
  int v41; // edx
  int v42; // r12d
  int v43; // eax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  __int64 v46; // r12
  __int64 *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rax
  bool v51; // of
  __int64 v52; // r12
  __int64 *v53; // rax
  __int64 v54; // [rsp-B8h] [rbp-B8h]
  __int64 **v55; // [rsp-B0h] [rbp-B0h]
  int v56; // [rsp-A8h] [rbp-A8h]
  __int64 **v57; // [rsp-A0h] [rbp-A0h]
  __int64 v58; // [rsp-A0h] [rbp-A0h]
  __int64 v59; // [rsp-90h] [rbp-90h]
  int v60; // [rsp-90h] [rbp-90h]
  __int64 v61; // [rsp-88h] [rbp-88h]
  __int64 v62; // [rsp-80h] [rbp-80h]
  __int64 v63; // [rsp-80h] [rbp-80h]
  int v64; // [rsp-78h] [rbp-78h]
  int v65; // [rsp-74h] [rbp-74h]
  __int64 v66; // [rsp-70h] [rbp-70h]
  unsigned __int8 **v67; // [rsp-68h] [rbp-68h] BYREF
  __int64 v68; // [rsp-60h] [rbp-60h]
  _BYTE v69[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_BYTE *)(a1 + 1800) )
    return 0;
  v2 = a1;
  v3 = *(_QWORD *)a1;
  if ( !v3 || (v59 = v3 + 48, *(_QWORD *)(v3 + 56) == v3 + 48) )
  {
    v66 = 0;
  }
  else
  {
    v66 = 0;
    v65 = 0;
    v62 = v2;
    v4 = *(_QWORD *)(v3 + 56);
    while ( 1 )
    {
      v5 = v4 - 24;
      if ( !v4 )
        v5 = 0;
      if ( v5 != sub_986580(v3) )
      {
        v7 = *(__int64 ***)(v62 + 48);
        v8 = *(unsigned int *)(v62 + 1824);
        v9 = 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
        {
          v10 = *(_QWORD *)(v5 - 8);
          v11 = v10 + v9;
        }
        else
        {
          v10 = v5 - v9;
          v11 = v5;
        }
        v12 = v11 - v10;
        v68 = 0x400000000LL;
        v13 = v12 >> 5;
        v67 = (unsigned __int8 **)v69;
        v14 = v12 >> 5;
        if ( (unsigned __int64)v12 > 0x80 )
        {
          v56 = v8;
          v57 = v7;
          sub_C8D5F0((__int64)&v67, v69, v12 >> 5, 8u, v8, v6);
          v17 = v67;
          v16 = v68;
          LODWORD(v13) = v12 >> 5;
          v7 = v57;
          LODWORD(v8) = v56;
          v15 = &v67[(unsigned int)v68];
        }
        else
        {
          v15 = (unsigned __int8 **)v69;
          v16 = 0;
          v17 = (unsigned __int8 **)v69;
        }
        if ( v12 > 0 )
        {
          v18 = 0;
          do
          {
            v15[v18 / 8] = *(unsigned __int8 **)(v10 + 4 * v18);
            v18 += 8LL;
            --v14;
          }
          while ( v14 );
          v17 = v67;
          v16 = v68;
        }
        LODWORD(v68) = v16 + v13;
        v19 = sub_DFCEF0(v7, (unsigned __int8 *)v5, v17, (unsigned int)(v16 + v13), v8);
        v21 = v20;
        if ( v67 != (unsigned __int8 **)v69 )
          _libc_free((unsigned __int64)v67);
        v22 = 1;
        if ( v21 != 1 )
          v22 = v65;
        v65 = v22;
        v23 = v19 + v66;
        if ( __OFADD__(v19, v66) )
        {
          v23 = 0x8000000000000000LL;
          if ( v19 > 0 )
            v23 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v66 = v23;
      }
      v4 = *(_QWORD *)(v4 + 8);
      if ( v59 == v4 )
        break;
      v3 = *(_QWORD *)v62;
    }
    v2 = v62;
  }
  v24 = *(_QWORD *)(v2 + 16);
  if ( v24 )
  {
    v58 = v24 + 48;
    if ( v24 + 48 == *(_QWORD *)(v24 + 56) )
    {
      if ( *(_QWORD *)(v2 + 1808) )
      {
        v52 = *(_QWORD *)(v2 + 928);
        v53 = sub_DD8400(v52, *(_QWORD *)(v2 + 24));
        if ( sub_DADE90(v52, (__int64)v53, *(_QWORD *)(v2 + 1808)) )
        {
          v67 = (unsigned __int8 **)sub_2AA7EC0(*(_QWORD *)(v2 + 1816), *(char **)(v2 + 1808), 0);
          if ( !BYTE4(v67) )
          {
            v61 = 0;
            v49 = 2;
LABEL_68:
            v54 = v61 / v49;
LABEL_62:
            v50 = 1;
            if ( v54 > 0 )
              v50 = v54;
            v51 = __OFADD__(v66, v50);
            v45 = v66 + v50;
            v66 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v51 )
              return v66;
            return v45;
          }
          v60 = 0;
          v61 = 0;
LABEL_60:
          v49 = (unsigned int)v67;
LABEL_61:
          if ( v60 )
            goto LABEL_62;
          goto LABEL_68;
        }
      }
      return v66;
    }
    else
    {
      v61 = 0;
      v60 = 0;
      v63 = v2;
      v25 = *(_QWORD *)(v24 + 56);
      while ( 1 )
      {
        v26 = v25 - 24;
        if ( !v25 )
          v26 = 0;
        if ( v26 != sub_986580(v24) )
        {
          v28 = *(__int64 ***)(v63 + 48);
          v29 = *(unsigned int *)(v63 + 1824);
          v30 = 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
          {
            v31 = *(_QWORD *)(v26 - 8);
            v32 = v31 + v30;
          }
          else
          {
            v31 = v26 - v30;
            v32 = v26;
          }
          v33 = v32 - v31;
          v68 = 0x400000000LL;
          v34 = v33 >> 5;
          v67 = (unsigned __int8 **)v69;
          v35 = v33 >> 5;
          if ( (unsigned __int64)v33 > 0x80 )
          {
            v64 = v29;
            v55 = v28;
            sub_C8D5F0((__int64)&v67, v69, v33 >> 5, 8u, v29, v27);
            v38 = v67;
            v37 = v68;
            LODWORD(v34) = v33 >> 5;
            v28 = v55;
            LODWORD(v29) = v64;
            v36 = &v67[(unsigned int)v68];
          }
          else
          {
            v36 = (unsigned __int8 **)v69;
            v37 = 0;
            v38 = (unsigned __int8 **)v69;
          }
          if ( v33 > 0 )
          {
            v39 = 0;
            do
            {
              v36[v39 / 8] = *(unsigned __int8 **)(v31 + 4 * v39);
              v39 += 8LL;
              --v35;
            }
            while ( v35 );
            v38 = v67;
            v37 = v68;
          }
          LODWORD(v68) = v34 + v37;
          v40 = sub_DFCEF0(v28, (unsigned __int8 *)v26, v38, (unsigned int)(v34 + v37), v29);
          v42 = v41;
          if ( v67 != (unsigned __int8 **)v69 )
            _libc_free((unsigned __int64)v67);
          v43 = 1;
          if ( v42 != 1 )
            v43 = v60;
          v60 = v43;
          v44 = v40 + v61;
          if ( __OFADD__(v40, v61) )
          {
            v44 = 0x8000000000000000LL;
            if ( v40 > 0 )
              v44 = 0x7FFFFFFFFFFFFFFFLL;
          }
          v61 = v44;
        }
        v25 = *(_QWORD *)(v25 + 8);
        if ( v58 == v25 )
          break;
        v24 = *(_QWORD *)(v63 + 16);
      }
      if ( *(_QWORD *)(v63 + 1808) )
      {
        v46 = *(_QWORD *)(v63 + 928);
        v47 = sub_DD8400(v46, *(_QWORD *)(v63 + 24));
        if ( sub_DADE90(v46, (__int64)v47, *(_QWORD *)(v63 + 1808)) )
        {
          v48 = sub_2AA7EC0(*(_QWORD *)(v63 + 1816), *(char **)(v63 + 1808), 0);
          v49 = 2;
          v67 = (unsigned __int8 **)v48;
          if ( !BYTE4(v48) )
            goto LABEL_61;
          goto LABEL_60;
        }
      }
      v45 = v61 + v66;
      if ( __OFADD__(v61, v66) )
      {
        v45 = 0x8000000000000000LL;
        if ( v61 > 0 )
          return 0x7FFFFFFFFFFFFFFFLL;
      }
    }
    return v45;
  }
  return v66;
}
