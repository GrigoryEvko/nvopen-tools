// Function: sub_11E8530
// Address: 0x11e8530
//
__int64 __fastcall sub_11E8530(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  _QWORD *v6; // r13
  int v8; // ecx
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v18; // edx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  char v26; // al
  __int64 v27; // r14
  _BYTE *v28; // r14
  _BYTE *v29; // rax
  _QWORD *v30; // rdi
  __int64 **v31; // r12
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v35; // r14
  __int64 v36; // rax
  char v37; // al
  char v38; // r12
  _QWORD *v39; // rax
  __int64 v40; // r9
  __int64 v41; // r13
  unsigned int *v42; // r14
  __int64 v43; // r12
  __int64 v44; // rdx
  unsigned int v45; // esi
  _QWORD *v46; // rdi
  __int64 v47; // rax
  _BYTE *v48; // rax
  _QWORD *v49; // rdi
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // r13
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // r9
  __int64 v57; // r12
  __int64 v58; // r13
  unsigned int *v59; // rbx
  unsigned int *v60; // r13
  __int64 v61; // rdx
  unsigned int v62; // esi
  unsigned int *v63; // r13
  __int64 v64; // r12
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // [rsp-10h] [rbp-D0h]
  __int64 v68; // [rsp+8h] [rbp-B8h]
  int v69; // [rsp+8h] [rbp-B8h]
  __int64 v70; // [rsp+10h] [rbp-B0h]
  char v71; // [rsp+10h] [rbp-B0h]
  __int64 v72; // [rsp+18h] [rbp-A8h]
  __int64 v73; // [rsp+18h] [rbp-A8h]
  int v74; // [rsp+18h] [rbp-A8h]
  void *s; // [rsp+20h] [rbp-A0h] BYREF
  size_t n; // [rsp+28h] [rbp-98h]
  _BYTE *v77[4]; // [rsp+30h] [rbp-90h] BYREF
  char v78; // [rsp+50h] [rbp-70h]
  char v79; // [rsp+51h] [rbp-6Fh]
  const char *v80; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v81; // [rsp+68h] [rbp-58h]
  __int16 v82; // [rsp+80h] [rbp-40h]

  v4 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v5 = *(_QWORD *)&a2[32 * (1 - v4)];
  if ( *(_BYTE *)v5 != 17 )
    return 0;
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  v8 = *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL);
  v9 = 0;
  if ( v8 )
    v9 = (1LL << ((unsigned __int8)v8 - 1)) - 1;
  if ( v9 < (unsigned __int64)v6 )
    return 0;
  s = 0;
  n = 0;
  v70 = *(_QWORD *)&a2[-32 * v4];
  v72 = *(_QWORD *)&a2[32 * (2 - v4)];
  if ( !(unsigned __int8)sub_98B0F0(v72, &s, 1u) )
    return 0;
  v10 = *a2;
  if ( v10 == 40 )
  {
    v11 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_17;
  }
  else
  {
    v11 = 0;
    if ( v10 != 85 )
    {
      v11 = 64;
      if ( v10 != 34 )
LABEL_74:
        BUG();
    }
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_17;
  }
  v12 = sub_BD2BC0((__int64)a2);
  v68 = v13 + v12;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v68 >> 4) )
LABEL_76:
      BUG();
LABEL_17:
    v16 = 0;
    goto LABEL_18;
  }
  if ( !(unsigned int)((v68 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_17;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_76;
  v69 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v14 = sub_BD2BC0((__int64)a2);
  v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v69);
LABEL_18:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v11 - v16) >> 5) != 3 )
  {
    if ( n != 2 || *(_BYTE *)s != 37 )
      return 0;
    v18 = *a2;
    if ( v18 == 40 )
    {
      v19 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v19 = 0;
      if ( v18 != 85 )
      {
        v19 = 64;
        if ( v18 != 34 )
          goto LABEL_74;
      }
    }
    if ( (a2[7] & 0x80u) != 0 )
    {
      v20 = sub_BD2BC0((__int64)a2);
      v73 = v21 + v20;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v73 >> 4) )
          goto LABEL_78;
      }
      else if ( (unsigned int)((v73 - sub_BD2BC0((__int64)a2)) >> 4) )
      {
        if ( (a2[7] & 0x80u) != 0 )
        {
          v74 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v22 = sub_BD2BC0((__int64)a2);
          v24 = 32LL * (unsigned int)(*(_DWORD *)(v22 + v23 - 4) - v74);
LABEL_36:
          v25 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
          if ( (unsigned int)((32 * v25 - 32 - v19 - v24) >> 5) != 4 )
            return 0;
          v26 = *((_BYTE *)s + 1);
          if ( v26 != 99 )
          {
            if ( v26 == 115 )
            {
              v80 = 0;
              v81 = 0;
              v27 = *(_QWORD *)&a2[32 * (3 - v25)];
              if ( (unsigned __int8)sub_98B0F0(v27, &v80, 1u) )
                return sub_11E8250(a1, (__int64)a2, v27, (__int64)v80, v81, (unsigned __int64)v6, a3);
            }
            return 0;
          }
          if ( (unsigned __int64)v6 <= 1 )
            return sub_11E8250(a1, (__int64)a2, 0, (__int64)"*", 1u, (unsigned __int64)v6, a3);
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a2[32 * (3 - v25)] + 8LL) + 8LL) != 12 )
            return 0;
          v30 = *(_QWORD **)(a3 + 72);
          v79 = 1;
          v77[0] = "char";
          v78 = 3;
          v31 = (__int64 **)sub_BCB2B0(v30);
          v32 = *(_QWORD *)&a2[32 * (3LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
          if ( v31 == *(__int64 ***)(v32 + 8) )
          {
            v35 = *(_QWORD *)&a2[32 * (3LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
            goto LABEL_55;
          }
          v33 = *(_QWORD *)(a3 + 80);
          v34 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v33 + 120LL);
          if ( v34 == sub_920130 )
          {
            if ( *(_BYTE *)v32 > 0x15u )
              goto LABEL_66;
            if ( (unsigned __int8)sub_AC4810(0x26u) )
              v35 = sub_ADAB70(38, v32, v31, 0);
            else
              v35 = sub_AA93C0(0x26u, v32, (__int64)v31);
          }
          else
          {
            v35 = v34(v33, 38u, (_BYTE *)v32, (__int64)v31);
          }
          if ( v35 )
          {
LABEL_55:
            v36 = sub_AA4E30(*(_QWORD *)(a3 + 48));
            v37 = sub_AE5020(v36, *(_QWORD *)(v35 + 8));
            v82 = 257;
            v38 = v37;
            v39 = sub_BD2C40(80, unk_3F10A10);
            v41 = (__int64)v39;
            if ( v39 )
              sub_B4D3C0((__int64)v39, v35, v70, 0, v38, v40, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
              *(_QWORD *)(a3 + 88),
              v41,
              &v80,
              *(_QWORD *)(a3 + 56),
              *(_QWORD *)(a3 + 64));
            v42 = *(unsigned int **)a3;
            v43 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
            if ( *(_QWORD *)a3 != v43 )
            {
              do
              {
                v44 = *((_QWORD *)v42 + 1);
                v45 = *v42;
                v42 += 4;
                sub_B99FD0(v41, v45, v44);
              }
              while ( (unsigned int *)v43 != v42 );
            }
            v46 = *(_QWORD **)(a3 + 72);
            v80 = "nul";
            v82 = 259;
            v47 = sub_BCB2D0(v46);
            v48 = (_BYTE *)sub_ACD640(v47, 1, 0);
            v49 = *(_QWORD **)(a3 + 72);
            v77[0] = v48;
            v50 = sub_BCB2B0(v49);
            v51 = sub_921130((unsigned int **)a3, v50, v70, v77, 1, (__int64)&v80, 3u);
            v52 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
            v53 = sub_ACD640(v52, 0, 0);
            v54 = sub_AA4E30(*(_QWORD *)(a3 + 48));
            v71 = sub_AE5020(v54, *(_QWORD *)(v53 + 8));
            v82 = 257;
            v55 = sub_BD2C40(80, unk_3F10A10);
            v56 = v67;
            v57 = (__int64)v55;
            if ( v55 )
              sub_B4D3C0((__int64)v55, v53, v51, 0, v71, v67, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88)
                                                                                            + 16LL))(
              *(_QWORD *)(a3 + 88),
              v57,
              &v80,
              *(_QWORD *)(a3 + 56),
              *(_QWORD *)(a3 + 64),
              v56);
            v58 = 4LL * *(unsigned int *)(a3 + 8);
            v59 = *(unsigned int **)a3;
            v60 = &v59[v58];
            while ( v60 != v59 )
            {
              v61 = *((_QWORD *)v59 + 1);
              v62 = *v59;
              v59 += 4;
              sub_B99FD0(v57, v62, v61);
            }
            return sub_AD64C0(*((_QWORD *)a2 + 1), 1, 0);
          }
LABEL_66:
          v82 = 257;
          v35 = sub_B51D30(38, v32, (__int64)v31, (__int64)&v80, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v35,
            v77,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v63 = *(unsigned int **)a3;
          v64 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v64 )
          {
            do
            {
              v65 = *((_QWORD *)v63 + 1);
              v66 = *v63;
              v63 += 4;
              sub_B99FD0(v35, v66, v65);
            }
            while ( (unsigned int *)v64 != v63 );
          }
          goto LABEL_55;
        }
LABEL_78:
        BUG();
      }
    }
    v24 = 0;
    goto LABEL_36;
  }
  if ( n )
  {
    v28 = s;
    v29 = memchr(s, 37, n);
    if ( v29 )
    {
      if ( v29 - v28 != -1 )
        return 0;
    }
  }
  return sub_11E8250(a1, (__int64)a2, v72, (__int64)s, n, (unsigned __int64)v6, a3);
}
