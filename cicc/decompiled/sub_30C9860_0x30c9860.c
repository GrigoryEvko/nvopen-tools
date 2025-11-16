// Function: sub_30C9860
// Address: 0x30c9860
//
__int64 __fastcall sub_30C9860(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 *v4; // r12
  __int64 *v5; // r13
  unsigned __int64 v6; // r14
  _QWORD *v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  int v10; // eax
  unsigned int v11; // r12d
  int v12; // r14d
  __int64 v13; // r8
  int v14; // r10d
  _QWORD *v15; // r9
  unsigned int v16; // r15d
  unsigned int v17; // ecx
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned int v22; // eax
  __int64 v23; // rsi
  int v24; // edx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // r12
  unsigned __int64 v28; // rax
  __int64 v29; // r15
  int v30; // r14d
  unsigned int v31; // r13d
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // r9
  int v35; // r11d
  _QWORD *v36; // r10
  unsigned int v37; // r12d
  unsigned int v38; // edi
  _QWORD *v39; // rcx
  __int64 v40; // rdx
  int v41; // ecx
  __int64 v42; // rdx
  unsigned __int64 v43; // r8
  unsigned int v44; // r15d
  _QWORD *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v49; // rdx
  __int64 v50; // rdi
  int v51; // r11d
  unsigned int v52; // r12d
  _QWORD *v53; // r8
  __int64 v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rdx
  int v57; // r10d
  unsigned int v58; // r11d
  __int64 v59; // [rsp+8h] [rbp-D8h]
  __int64 v61; // [rsp+18h] [rbp-C8h]
  __int64 v62; // [rsp+18h] [rbp-C8h]
  __int64 v63; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v64; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v65; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v67; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v68; // [rsp+38h] [rbp-A8h]
  _QWORD *v69; // [rsp+48h] [rbp-98h] BYREF
  __int64 v70; // [rsp+50h] [rbp-90h] BYREF
  __int64 v71; // [rsp+58h] [rbp-88h]
  __int64 v72; // [rsp+60h] [rbp-80h]
  __int64 v73; // [rsp+68h] [rbp-78h]
  __int64 *v74; // [rsp+70h] [rbp-70h] BYREF
  __int64 v75; // [rsp+78h] [rbp-68h]
  __int64 v76; // [rsp+80h] [rbp-60h] BYREF
  __int64 v77; // [rsp+88h] [rbp-58h]
  __int64 v78; // [rsp+90h] [rbp-50h]
  __int64 v79; // [rsp+98h] [rbp-48h]
  _BYTE *v80; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v81; // [rsp+A8h] [rbp-38h]
  _BYTE v82[48]; // [rsp+B0h] [rbp-30h] BYREF

  v74 = &v76;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = v82;
  v81 = 0;
  v2 = sub_30C9020((__int64)a1, a2);
  v3 = a1[2][10];
  if ( v3 )
    v3 -= 24;
  if ( a1[1] != (_QWORD *)v3 )
  {
    v69 = (_QWORD *)v3;
    sub_30C6060((__int64)&v70, (__int64 *)&v69);
  }
  v4 = a1[4];
  v5 = &v4[*((unsigned int *)a1 + 12)];
  if ( *((_DWORD *)a1 + 10) && v4 != v5 )
  {
    while ( *v4 == -4096 || *v4 == -8192 )
    {
      if ( ++v4 == v5 )
        goto LABEL_6;
    }
    if ( v4 != v5 )
    {
      v55 = *v4;
      v69 = (_QWORD *)v55;
      if ( !v55 )
        goto LABEL_106;
LABEL_98:
      v56 = (unsigned int)(*(_DWORD *)(v55 + 44) + 1);
      if ( (unsigned int)(*(_DWORD *)(v55 + 44) + 1) < *(_DWORD *)(v2 + 32) )
      {
LABEL_99:
        if ( *(_QWORD *)(*(_QWORD *)(v2 + 24) + 8 * v56) )
        {
          sub_30C6060((__int64)&v70, (__int64 *)&v69);
          goto LABEL_101;
        }
      }
LABEL_107:
      sub_30C6060((__int64)&v76, (__int64 *)&v69);
LABEL_101:
      while ( ++v4 != v5 )
      {
        if ( *v4 != -8192 && *v4 != -4096 )
        {
          if ( v4 == v5 )
            break;
          v55 = *v4;
          v69 = (_QWORD *)v55;
          if ( v55 )
            goto LABEL_98;
LABEL_106:
          v56 = 0;
          if ( *(_DWORD *)(v2 + 32) )
            goto LABEL_99;
          goto LABEL_107;
        }
      }
    }
  }
LABEL_6:
  v6 = 0;
  v67 = (unsigned int)v75;
  v69 = a1[1];
  sub_30C6060((__int64)&v70, (__int64 *)&v69);
  if ( !(_DWORD)v75 )
    goto LABEL_28;
  v59 = v2;
  do
  {
    while ( 1 )
    {
      v7 = (_QWORD *)v74[v6];
      sub_30C4100(*a1, v7);
      if ( v67 <= v6 )
      {
        v8 = v7[6] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_QWORD *)v8 != v7 + 6 )
        {
          if ( !v8 )
LABEL_70:
            BUG();
          v9 = v8 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 <= 0xA )
          {
            v10 = sub_B46E30(v9);
            if ( v10 )
              break;
          }
        }
      }
      if ( ++v6 >= (unsigned int)v75 )
        goto LABEL_27;
    }
    v64 = v6;
    v11 = 0;
    v12 = v10;
    do
    {
      while ( 1 )
      {
        v20 = sub_B46EC0(v9, v11);
        v21 = v20;
        if ( !(_DWORD)v73 )
        {
          ++v70;
LABEL_19:
          sub_E3B4A0((__int64)&v70, 2 * v73);
          if ( !(_DWORD)v73 )
            goto LABEL_140;
          v22 = (v73 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v15 = (_QWORD *)(v71 + 8LL * v22);
          v23 = *v15;
          v24 = v72 + 1;
          if ( v21 != *v15 )
          {
            v57 = 1;
            v13 = 0;
            while ( v23 != -4096 )
            {
              if ( v23 == -8192 && !v13 )
                v13 = (__int64)v15;
              v22 = (v73 - 1) & (v57 + v22);
              v15 = (_QWORD *)(v71 + 8LL * v22);
              v23 = *v15;
              if ( v21 == *v15 )
                goto LABEL_21;
              ++v57;
            }
            if ( v13 )
              v15 = (_QWORD *)v13;
          }
          goto LABEL_21;
        }
        v13 = (unsigned int)(v73 - 1);
        v14 = 1;
        v15 = 0;
        v16 = ((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4);
        v17 = v13 & v16;
        v18 = (_QWORD *)(v71 + 8LL * ((unsigned int)v13 & v16));
        v19 = *v18;
        if ( v21 != *v18 )
          break;
LABEL_16:
        if ( v12 == ++v11 )
          goto LABEL_26;
      }
      while ( v19 != -4096 )
      {
        if ( v15 || v19 != -8192 )
          v18 = v15;
        v17 = v13 & (v14 + v17);
        v19 = *(_QWORD *)(v71 + 8LL * v17);
        if ( v21 == v19 )
          goto LABEL_16;
        ++v14;
        v15 = v18;
        v18 = (_QWORD *)(v71 + 8LL * v17);
      }
      if ( !v15 )
        v15 = v18;
      ++v70;
      v24 = v72 + 1;
      if ( 4 * ((int)v72 + 1) >= (unsigned int)(3 * v73) )
        goto LABEL_19;
      if ( (int)v73 - HIDWORD(v72) - v24 <= (unsigned int)v73 >> 3 )
      {
        sub_E3B4A0((__int64)&v70, v73);
        if ( !(_DWORD)v73 )
        {
LABEL_140:
          LODWORD(v72) = v72 + 1;
          BUG();
        }
        v13 = 1;
        v44 = (v73 - 1) & v16;
        v15 = (_QWORD *)(v71 + 8LL * v44);
        v24 = v72 + 1;
        v45 = 0;
        v46 = *v15;
        if ( v21 != *v15 )
        {
          while ( v46 != -4096 )
          {
            if ( v46 == -8192 && !v45 )
              v45 = v15;
            v44 = (v73 - 1) & (v13 + v44);
            v15 = (_QWORD *)(v71 + 8LL * v44);
            v46 = *v15;
            if ( v21 == *v15 )
              goto LABEL_21;
            v13 = (unsigned int)(v13 + 1);
          }
          if ( v45 )
            v15 = v45;
        }
      }
LABEL_21:
      LODWORD(v72) = v24;
      if ( *v15 != -4096 )
        --HIDWORD(v72);
      *v15 = v21;
      v25 = (unsigned int)v75;
      v26 = (unsigned int)v75 + 1LL;
      if ( v26 > HIDWORD(v75) )
      {
        sub_C8D5F0((__int64)&v74, &v76, v26, 8u, v13, (__int64)v15);
        v25 = (unsigned int)v75;
      }
      ++v11;
      v74[v25] = v21;
      LODWORD(v75) = v75 + 1;
    }
    while ( v12 != v11 );
LABEL_26:
    v6 = v64 + 1;
  }
  while ( v64 + 1 < (unsigned int)v75 );
LABEL_27:
  v2 = v59;
LABEL_28:
  v65 = (unsigned int)v81;
  if ( (_DWORD)v81 )
  {
    v68 = 0;
    v27 = *(_QWORD *)v80;
    while ( 1 )
    {
      v28 = *(_QWORD *)(v27 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v28 != v27 + 48 )
      {
        if ( !v28 )
          goto LABEL_70;
        v29 = v28 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 <= 0xA )
        {
          v30 = sub_B46E30(v29);
          if ( v30 )
            break;
        }
      }
LABEL_52:
      if ( ++v68 >= (unsigned int)v81 )
        goto LABEL_71;
      v27 = *(_QWORD *)&v80[8 * v68];
      if ( v68 >= v65 )
        sub_30C3570(*a1, *(_QWORD **)&v80[8 * v68], -1);
    }
    v31 = 0;
    while ( 2 )
    {
      v33 = sub_B46EC0(v29, v31);
      if ( v33 )
      {
        v32 = (unsigned int)(*(_DWORD *)(v33 + 44) + 1);
        if ( (unsigned int)(*(_DWORD *)(v33 + 44) + 1) >= *(_DWORD *)(v2 + 32) )
          goto LABEL_40;
      }
      else
      {
        v32 = 0;
        if ( !*(_DWORD *)(v2 + 32) )
          goto LABEL_40;
      }
      if ( *(_QWORD *)(*(_QWORD *)(v2 + 24) + 8 * v32) )
        goto LABEL_37;
LABEL_40:
      if ( (_DWORD)v79 )
      {
        v34 = (unsigned int)(v79 - 1);
        v35 = 1;
        v36 = 0;
        v37 = ((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4);
        v38 = v34 & v37;
        v39 = (_QWORD *)(v77 + 8LL * ((unsigned int)v34 & v37));
        v40 = *v39;
        if ( v33 == *v39 )
        {
LABEL_37:
          if ( v30 == ++v31 )
            goto LABEL_52;
          continue;
        }
        while ( v40 != -4096 )
        {
          if ( v40 != -8192 || v36 )
            v39 = v36;
          v38 = v34 & (v35 + v38);
          v40 = *(_QWORD *)(v77 + 8LL * v38);
          if ( v33 == v40 )
            goto LABEL_37;
          ++v35;
          v36 = v39;
          v39 = (_QWORD *)(v77 + 8LL * v38);
        }
        if ( !v36 )
          v36 = v39;
        ++v76;
        v41 = v78 + 1;
        if ( 4 * ((int)v78 + 1) < (unsigned int)(3 * v79) )
        {
          if ( (int)v79 - HIDWORD(v78) - v41 <= (unsigned int)v79 >> 3 )
          {
            v63 = v33;
            sub_E3B4A0((__int64)&v76, v79);
            if ( !(_DWORD)v79 )
            {
LABEL_139:
              LODWORD(v78) = v78 + 1;
              BUG();
            }
            v34 = 1;
            v52 = (v79 - 1) & v37;
            v53 = 0;
            v36 = (_QWORD *)(v77 + 8LL * v52);
            v41 = v78 + 1;
            v33 = v63;
            v54 = *v36;
            if ( v63 != *v36 )
            {
              while ( v54 != -4096 )
              {
                if ( !v53 && v54 == -8192 )
                  v53 = v36;
                v58 = v34 + 1;
                v34 = ((_DWORD)v79 - 1) & (v52 + (unsigned int)v34);
                v36 = (_QWORD *)(v77 + 8LL * (unsigned int)v34);
                v52 = v34;
                v54 = *v36;
                if ( v63 == *v36 )
                  goto LABEL_47;
                v34 = v58;
              }
              if ( v53 )
                v36 = v53;
            }
          }
LABEL_47:
          LODWORD(v78) = v41;
          if ( *v36 != -4096 )
            --HIDWORD(v78);
          *v36 = v33;
          v42 = (unsigned int)v81;
          v43 = (unsigned int)v81 + 1LL;
          if ( v43 > HIDWORD(v81) )
          {
            v62 = v33;
            sub_C8D5F0((__int64)&v80, v82, (unsigned int)v81 + 1LL, 8u, v43, v34);
            v42 = (unsigned int)v81;
            v33 = v62;
          }
          ++v31;
          *(_QWORD *)&v80[8 * v42] = v33;
          LODWORD(v81) = v81 + 1;
          if ( v30 == v31 )
            goto LABEL_52;
          continue;
        }
      }
      else
      {
        ++v76;
      }
      break;
    }
    v61 = v33;
    sub_E3B4A0((__int64)&v76, 2 * v79);
    if ( !(_DWORD)v79 )
      goto LABEL_139;
    v33 = v61;
    LODWORD(v49) = (v79 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
    v36 = (_QWORD *)(v77 + 8LL * (unsigned int)v49);
    v50 = *v36;
    v41 = v78 + 1;
    if ( v61 != *v36 )
    {
      v51 = 1;
      v34 = 0;
      while ( v50 != -4096 )
      {
        if ( v50 == -8192 && !v34 )
          v34 = (__int64)v36;
        v49 = ((_DWORD)v79 - 1) & (unsigned int)(v49 + v51);
        v36 = (_QWORD *)(v77 + 8 * v49);
        v50 = *v36;
        if ( v61 == *v36 )
          goto LABEL_47;
        ++v51;
      }
      if ( v34 )
        v36 = (_QWORD *)v34;
    }
    goto LABEL_47;
  }
LABEL_71:
  v47 = sub_BC1CD0(a2, &unk_4F875F0, (__int64)a1[2]);
  sub_30C8C60(*a1, (__int64)a1[2], v47 + 8);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  sub_C7D6A0(v77, 8LL * (unsigned int)v79, 8);
  if ( v74 != &v76 )
    _libc_free((unsigned __int64)v74);
  return sub_C7D6A0(v71, 8LL * (unsigned int)v73, 8);
}
