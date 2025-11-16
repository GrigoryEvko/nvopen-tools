// Function: sub_D738B0
// Address: 0xd738b0
//
void __fastcall sub_D738B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // r12
  char v7; // r13
  char *v8; // r8
  __int64 v9; // rdi
  char *v10; // rdx
  char *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  char *v14; // r15
  unsigned __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  _QWORD *v20; // r14
  _QWORD *v21; // r13
  _QWORD *v22; // r15
  _QWORD *v23; // rax
  bool v24; // al
  unsigned __int64 v25; // rax
  __int64 v26; // r13
  unsigned int i; // r15d
  unsigned __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  int v34; // edx
  __int64 v35; // rdi
  __int64 v36; // r14
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  __int64 v39; // r15
  int v40; // eax
  unsigned int v41; // r13d
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // r10
  __int64 v46; // rdi
  __int64 v47; // r8
  __int64 v48; // r9
  char *v49; // rax
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  char v52; // dl
  int v53; // eax
  int v54; // edi
  __int64 v55; // r12
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 v58; // rdx
  __int64 v59; // rdx
  _QWORD *v60; // rdi
  __int64 v61; // r12
  __int64 v62; // rax
  __int64 v63; // [rsp+8h] [rbp-178h]
  __int64 v64; // [rsp+8h] [rbp-178h]
  unsigned __int64 v65; // [rsp+18h] [rbp-168h]
  __int64 v66; // [rsp+28h] [rbp-158h]
  __int64 v67; // [rsp+28h] [rbp-158h]
  unsigned __int64 v68; // [rsp+30h] [rbp-150h]
  __int64 v69; // [rsp+38h] [rbp-148h]
  int v70; // [rsp+38h] [rbp-148h]
  int v71; // [rsp+38h] [rbp-148h]
  int v72; // [rsp+38h] [rbp-148h]
  _QWORD v73[2]; // [rsp+40h] [rbp-140h] BYREF
  unsigned __int64 v74; // [rsp+50h] [rbp-130h]
  __int64 v75; // [rsp+60h] [rbp-120h] BYREF
  char *v76; // [rsp+68h] [rbp-118h]
  __int64 v77; // [rsp+70h] [rbp-110h]
  int v78; // [rsp+78h] [rbp-108h]
  char v79; // [rsp+7Ch] [rbp-104h]
  char v80; // [rsp+80h] [rbp-100h] BYREF
  _BYTE *v81; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v82; // [rsp+C8h] [rbp-B8h]
  _BYTE v83[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v76 = &v80;
  v3 = *(_QWORD *)a2;
  v81 = v83;
  v82 = 0x1000000000LL;
  v4 = *(unsigned int *)(a2 + 8);
  v5 = v3;
  v68 = v3;
  v75 = 0;
  v77 = 8;
  v78 = 0;
  v79 = 1;
  v65 = v3 + 24 * v4;
  if ( v3 == v65 )
    return;
  while ( 1 )
  {
    v6 = *(_QWORD *)(v68 + 16);
    if ( v6 )
    {
      v7 = *(_BYTE *)v6;
      if ( (unsigned __int8)(*(_BYTE *)v6 - 26) <= 2u )
        break;
    }
LABEL_33:
    v68 += 24LL;
    if ( v65 == v68 )
    {
      if ( v81 != v83 )
        _libc_free(v81, v5);
      if ( v79 )
        return;
LABEL_91:
      _libc_free(v76, v5);
      return;
    }
  }
  v5 = *(_QWORD *)(v6 + 64);
  v69 = sub_D68C20(*(_QWORD *)a1, v5);
  if ( v7 == 28 )
  {
    v74 = v6;
    v73[0] = 0;
    v73[1] = 0;
    if ( v6 != -4096 && v6 != -8192 )
      sub_BD73F0((__int64)v73);
    if ( !*(_QWORD *)(a1 + 752) )
    {
      v8 = *(char **)(a1 + 504);
      v9 = *(unsigned int *)(a1 + 512);
      v10 = &v8[24 * v9];
      if ( v8 != v10 )
      {
        v11 = *(char **)(a1 + 504);
        while ( 1 )
        {
          v12 = *((_QWORD *)v11 + 2);
          if ( v12 == v74 )
            break;
          v11 += 24;
          if ( v10 == v11 )
            goto LABEL_27;
        }
        if ( v11 != v10 )
        {
          v13 = v10 - (v11 + 24);
          v5 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
          if ( v13 > 0 )
          {
            v63 = a1;
            v14 = v11 + 24;
            v15 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
            while ( 1 )
            {
              v16 = *((_QWORD *)v11 + 5);
              if ( v16 != v12 )
              {
                if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
                  sub_BD60C0(v11);
                *((_QWORD *)v11 + 2) = v16;
                if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
                  sub_BD73F0((__int64)v11);
              }
              v11 = v14;
              if ( !--v15 )
                break;
              v12 = *((_QWORD *)v14 + 2);
              v14 += 24;
            }
            a1 = v63;
            LODWORD(v9) = *(_DWORD *)(v63 + 512);
            v8 = *(char **)(v63 + 504);
          }
          v17 = (unsigned int)(v9 - 1);
          *(_DWORD *)(a1 + 512) = v17;
          sub_D68D70(&v8[24 * v17]);
        }
      }
      goto LABEL_27;
    }
    v20 = (_QWORD *)(a1 + 720);
    if ( *(_QWORD *)(a1 + 728) )
    {
      v21 = (_QWORD *)(a1 + 720);
      v22 = *(_QWORD **)(a1 + 728);
      while ( 1 )
      {
        while ( v22[6] < v74 )
        {
          v22 = (_QWORD *)v22[3];
          if ( !v22 )
            goto LABEL_44;
        }
        v23 = (_QWORD *)v22[2];
        if ( v22[6] <= v74 )
          break;
        v21 = v22;
        v22 = (_QWORD *)v22[2];
        if ( !v23 )
        {
LABEL_44:
          v24 = v20 == v21;
          goto LABEL_45;
        }
      }
      v5 = v22[3];
      if ( v5 )
      {
        do
        {
          while ( 1 )
          {
            v57 = *(_QWORD *)(v5 + 16);
            v58 = *(_QWORD *)(v5 + 24);
            if ( v74 < *(_QWORD *)(v5 + 48) )
              break;
            v5 = *(_QWORD *)(v5 + 24);
            if ( !v58 )
              goto LABEL_97;
          }
          v21 = (_QWORD *)v5;
          v5 = *(_QWORD *)(v5 + 16);
        }
        while ( v57 );
      }
LABEL_97:
      while ( v23 )
      {
        while ( 1 )
        {
          v5 = v23[2];
          v59 = v23[3];
          if ( v74 <= v23[6] )
            break;
          v23 = (_QWORD *)v23[3];
          if ( !v59 )
            goto LABEL_100;
        }
        v22 = v23;
        v23 = (_QWORD *)v23[2];
      }
LABEL_100:
      if ( *(_QWORD **)(a1 + 736) != v22 || v20 != v21 )
      {
        if ( v22 != v21 )
        {
          v64 = v6;
          do
          {
            v60 = v22;
            v22 = (_QWORD *)sub_220EF30(v22);
            v61 = sub_220F330(v60, a1 + 720);
            v62 = *(_QWORD *)(v61 + 48);
            if ( v62 != -4096 && v62 != 0 && v62 != -8192 )
              sub_BD60C0((_QWORD *)(v61 + 32));
            v5 = 56;
            j_j___libc_free_0(v61, 56);
            --*(_QWORD *)(a1 + 752);
          }
          while ( v22 != v21 );
          v6 = v64;
        }
        goto LABEL_27;
      }
    }
    else
    {
      v21 = (_QWORD *)(a1 + 720);
      v24 = 1;
LABEL_45:
      if ( *(_QWORD **)(a1 + 736) != v21 || !v24 )
        goto LABEL_27;
    }
    sub_D690E0(*(_QWORD **)(a1 + 728));
    *(_QWORD *)(a1 + 736) = v20;
    *(_QWORD *)(a1 + 728) = 0;
    *(_QWORD *)(a1 + 744) = v20;
    *(_QWORD *)(a1 + 752) = 0;
LABEL_27:
    sub_D68D70(v73);
  }
  v18 = *(_QWORD *)(v6 + 56);
  if ( v18 != v69 )
  {
    if ( !v18 )
      goto LABEL_113;
    v19 = v18 - 112;
    if ( *(_BYTE *)(v18 - 48) == 26 )
      v19 = v18 - 80;
    v5 = v6;
    sub_AC2B30(v19, v6);
    goto LABEL_33;
  }
  v25 = sub_986580(*(_QWORD *)(v6 + 64));
  v26 = v25;
  if ( v25 )
  {
    v70 = sub_B46E30(v25);
    if ( v70 )
    {
      for ( i = 0; i != v70; ++i )
      {
        while ( 1 )
        {
          v5 = sub_B46EC0(v26, i);
          v28 = v5;
          v29 = sub_D68B40(*(_QWORD *)a1, v5);
          if ( !v29 )
            break;
          v5 = *(_QWORD *)(v6 + 64);
          ++i;
          sub_D682A0(v29, v5, v6);
          if ( v70 == i )
            goto LABEL_56;
        }
        v32 = (unsigned int)v82;
        v33 = (unsigned int)v82 + 1LL;
        if ( v33 > HIDWORD(v82) )
        {
          v5 = (unsigned __int64)v83;
          sub_C8D5F0((__int64)&v81, v83, v33, 8u, v30, v31);
          v32 = (unsigned int)v82;
        }
        *(_QWORD *)&v81[8 * v32] = v28;
        LODWORD(v82) = v82 + 1;
      }
    }
  }
LABEL_56:
  v34 = v82;
  while ( 2 )
  {
    if ( !v34 )
      goto LABEL_33;
    v35 = *(_QWORD *)a1;
    v71 = v34 - 1;
    v36 = *(_QWORD *)&v81[8 * v34 - 8];
    LODWORD(v82) = v34 - 1;
    v5 = v36;
    v37 = sub_D68C20(v35, v36);
    v34 = v71;
    if ( !v37 )
    {
      v38 = *(_QWORD *)(v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v38 == v36 + 48 )
        continue;
      if ( !v38 )
        BUG();
      v39 = v38 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v38 - 24) - 30 > 0xA )
        continue;
      v40 = sub_B46E30(v39);
      v34 = v71;
      v72 = v40;
      if ( !v40 )
        continue;
      v41 = 0;
      while ( 2 )
      {
        v47 = sub_B46EC0(v39, v41);
        v42 = *(unsigned int *)(*(_QWORD *)a1 + 56LL);
        v5 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
        if ( !(_DWORD)v42 )
        {
LABEL_69:
          if ( !v79 )
            goto LABEL_78;
          v49 = v76;
          v42 = HIDWORD(v77);
          v43 = (__int64)&v76[8 * HIDWORD(v77)];
          if ( v76 != (char *)v43 )
          {
            while ( v47 != *(_QWORD *)v49 )
            {
              v49 += 8;
              if ( (char *)v43 == v49 )
                goto LABEL_73;
            }
            goto LABEL_67;
          }
LABEL_73:
          if ( HIDWORD(v77) < (unsigned int)v77 )
          {
            ++HIDWORD(v77);
            *(_QWORD *)v43 = v47;
            ++v75;
          }
          else
          {
LABEL_78:
            v5 = v47;
            v66 = v47;
            sub_C8CC70((__int64)&v75, v47, v43, v42, v47, v48);
            v47 = v66;
            if ( !v52 )
              goto LABEL_67;
          }
          v50 = (unsigned int)v82;
          v51 = (unsigned int)v82 + 1LL;
          if ( v51 > HIDWORD(v82) )
          {
            v5 = (unsigned __int64)v83;
            v67 = v47;
            sub_C8D5F0((__int64)&v81, v83, v51, 8u, v47, v48);
            v50 = (unsigned int)v82;
            v47 = v67;
          }
          *(_QWORD *)&v81[8 * v50] = v47;
          LODWORD(v82) = v82 + 1;
          goto LABEL_67;
        }
        v42 = (unsigned int)(v42 - 1);
        v43 = (unsigned int)v42 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v44 = (__int64 *)(v5 + 16 * v43);
        v45 = *v44;
        if ( v47 != *v44 )
        {
          v53 = 1;
          while ( v45 != -4096 )
          {
            v54 = v53 + 1;
            v43 = (unsigned int)v42 & (v53 + (_DWORD)v43);
            v44 = (__int64 *)(v5 + 16LL * (unsigned int)v43);
            v45 = *v44;
            if ( v47 == *v44 )
              goto LABEL_65;
            v53 = v54;
          }
          goto LABEL_69;
        }
LABEL_65:
        v46 = v44[1];
        if ( !v46 )
          goto LABEL_69;
        v5 = v36;
        sub_D682A0(v46, v36, v6);
LABEL_67:
        if ( ++v41 == v72 )
          goto LABEL_56;
        continue;
      }
    }
    break;
  }
  v55 = *(_QWORD *)(v37 + 8);
  if ( !v55 )
  {
    sub_D735C0((__int64 *)a1, 0);
LABEL_113:
    BUG();
  }
  v56 = v55 - 112;
  v5 = sub_D735C0((__int64 *)a1, v55 - 48);
  if ( *(_BYTE *)(v55 - 48) == 26 )
    v56 = v55 - 80;
  sub_AC2B30(v56, v5);
  if ( v81 != v83 )
    _libc_free(v81, v5);
  if ( !v79 )
    goto LABEL_91;
}
