// Function: sub_1AE4880
// Address: 0x1ae4880
//
__int64 __fastcall sub_1AE4880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 v5; // r12
  __int64 *v6; // r9
  int v7; // edx
  unsigned __int64 *v9; // rax
  __int64 *v10; // r8
  __int64 *v11; // rbx
  _BYTE *v12; // rdx
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r14
  _BYTE *v16; // rdi
  _BYTE *v17; // rdx
  __int64 **v18; // rcx
  __int64 *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned int v22; // ecx
  __int64 **v23; // rdx
  __int64 *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // r15
  _QWORD *v27; // rax
  _QWORD *v28; // rbx
  __int64 v29; // rax
  unsigned int v30; // esi
  _QWORD *v31; // rcx
  int v32; // esi
  __int64 v33; // rdx
  unsigned __int64 v34; // r13
  _BYTE *v35; // r14
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // rax
  unsigned int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rsi
  unsigned int v45; // edx
  __int64 *v46; // rdi
  unsigned int v47; // ecx
  unsigned int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  int v51; // r10d
  int v52; // r10d
  _QWORD *v53; // rdx
  unsigned int v54; // eax
  __int64 v55; // rsi
  int v56; // ecx
  __int64 *v57; // rdx
  int v58; // esi
  unsigned int v59; // eax
  __int64 v60; // rcx
  int v61; // edx
  __int64 v62; // [rsp+0h] [rbp-1B0h]
  _BYTE *v66; // [rsp+40h] [rbp-170h] BYREF
  __int64 v67; // [rsp+48h] [rbp-168h]
  _BYTE v68[64]; // [rsp+50h] [rbp-160h] BYREF
  _BYTE *v69; // [rsp+90h] [rbp-120h] BYREF
  __int64 v70; // [rsp+98h] [rbp-118h]
  _BYTE v71[64]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v72; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v73; // [rsp+E8h] [rbp-C8h]
  _QWORD *v74; // [rsp+F0h] [rbp-C0h] BYREF
  unsigned int v75; // [rsp+F8h] [rbp-B8h]
  _BYTE *v76; // [rsp+130h] [rbp-80h] BYREF
  __int64 v77; // [rsp+138h] [rbp-78h]
  _BYTE v78[112]; // [rsp+140h] [rbp-70h] BYREF

  v4 = 0;
  v5 = a1;
  v66 = v68;
  v67 = 0x800000000LL;
  sub_13F9EC0(a1, (__int64)&v66);
  v7 = v67;
  if ( !(_DWORD)v67 )
    goto LABEL_2;
  v9 = (unsigned __int64 *)&v74;
  v72 = 0;
  v73 = 1;
  do
    *v9++ = -8;
  while ( v9 != (unsigned __int64 *)&v76 );
  v76 = v78;
  v10 = (__int64 *)&v66[8 * v7];
  v11 = (__int64 *)(v66 + 8);
  v77 = 0x800000000LL;
  v12 = v71;
  v13 = v10;
  v70 = 0x800000000LL;
  v14 = 0;
  v69 = v71;
  v15 = *(_QWORD *)v66;
  while ( 1 )
  {
    *(_QWORD *)&v12[8 * v14] = v15;
    v14 = (unsigned int)(v70 + 1);
    LODWORD(v70) = v70 + 1;
    if ( v13 == v11 )
      break;
    v15 = *v11;
    if ( HIDWORD(v70) <= (unsigned int)v14 )
    {
      sub_16CD150((__int64)&v69, v71, 0, 8, (int)v10, (int)v6);
      v14 = (unsigned int)v70;
    }
    v12 = v69;
    ++v11;
  }
  v16 = v69;
LABEL_13:
  v17 = &v16[8 * (unsigned int)v14];
LABEL_14:
  if ( (_DWORD)v14 )
  {
    while ( 1 )
    {
      v18 = *(__int64 ***)(v5 + 32);
      LODWORD(v14) = v14 - 1;
      v19 = (__int64 *)*((_QWORD *)v17 - 1);
      v17 -= 8;
      LODWORD(v70) = v14;
      if ( v19 == *v18 )
        goto LABEL_14;
      v20 = *(unsigned int *)(a2 + 48);
      if ( !(_DWORD)v20 )
        goto LABEL_128;
      v21 = *(_QWORD *)(a2 + 32);
      LODWORD(v10) = v20 - 1;
      v22 = (v20 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v23 = (__int64 **)(v21 + 16LL * v22);
      v6 = *v23;
      if ( v19 != *v23 )
      {
        v50 = 1;
        while ( v6 != (__int64 *)-8LL )
        {
          v51 = v50 + 1;
          v22 = (unsigned int)v10 & (v50 + v22);
          v23 = (__int64 **)(v21 + 16LL * v22);
          v6 = *v23;
          if ( v19 == *v23 )
            goto LABEL_18;
          v50 = v51;
        }
LABEL_128:
        BUG();
      }
LABEL_18:
      if ( v23 == (__int64 **)(v21 + 16 * v20) )
        goto LABEL_128;
      v24 = v23[1];
      v25 = *(_QWORD **)(v5 + 72);
      v26 = *(_QWORD *)v24[1];
      v27 = *(_QWORD **)(v5 + 64);
      if ( v25 == v27 )
      {
        v28 = &v27[*(unsigned int *)(v5 + 84)];
        if ( v27 == v28 )
        {
          v53 = *(_QWORD **)(v5 + 64);
        }
        else
        {
          do
          {
            if ( v26 == *v27 )
              break;
            ++v27;
          }
          while ( v28 != v27 );
          v53 = v28;
        }
        goto LABEL_64;
      }
      v28 = &v25[*(unsigned int *)(v5 + 80)];
      v27 = sub_16CC9F0(v5 + 56, v26);
      if ( v26 == *v27 )
        break;
      v29 = *(_QWORD *)(v5 + 72);
      if ( v29 == *(_QWORD *)(v5 + 64) )
      {
        v27 = (_QWORD *)(v29 + 8LL * *(unsigned int *)(v5 + 84));
        v53 = v27;
LABEL_64:
        while ( v53 != v27 && *v27 >= 0xFFFFFFFFFFFFFFFELL )
          ++v27;
        goto LABEL_23;
      }
      v27 = (_QWORD *)(v29 + 8LL * *(unsigned int *)(v5 + 80));
LABEL_23:
      if ( v28 != v27 )
      {
        if ( (v73 & 1) != 0 )
        {
          v31 = &v74;
          v32 = 7;
        }
        else
        {
          v30 = v75;
          v31 = v74;
          if ( !v75 )
          {
            v45 = v73;
            ++v72;
            v46 = 0;
            v47 = ((unsigned int)v73 >> 1) + 1;
LABEL_72:
            v48 = 3 * v30;
            goto LABEL_73;
          }
          v32 = v75 - 1;
        }
        LODWORD(v33) = v32 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v6 = &v31[(unsigned int)v33];
        v10 = (__int64 *)*v6;
        if ( v26 != *v6 )
        {
          v52 = 1;
          v46 = 0;
          while ( v10 != (__int64 *)-8LL )
          {
            if ( v10 != (__int64 *)-16LL || v46 )
              v6 = v46;
            v33 = v32 & (unsigned int)(v33 + v52);
            v10 = (__int64 *)v31[v33];
            if ( (__int64 *)v26 == v10 )
              goto LABEL_28;
            ++v52;
            v46 = v6;
            v6 = &v31[v33];
          }
          v45 = v73;
          if ( !v46 )
            v46 = v6;
          ++v72;
          v47 = ((unsigned int)v73 >> 1) + 1;
          if ( (v73 & 1) == 0 )
          {
            v30 = v75;
            goto LABEL_72;
          }
          v48 = 24;
          v30 = 8;
LABEL_73:
          LODWORD(v10) = 4 * v47;
          if ( 4 * v47 < v48 )
          {
            if ( v30 - HIDWORD(v73) - v47 > v30 >> 3 )
              goto LABEL_75;
            sub_19628C0((__int64)&v72, v30);
            if ( (v73 & 1) != 0 )
            {
              v10 = (__int64 *)&v74;
              v58 = 7;
              goto LABEL_101;
            }
            v10 = v74;
            if ( v75 )
            {
              v58 = v75 - 1;
LABEL_101:
              v59 = v58 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
              v46 = &v10[v59];
              v45 = v73;
              v60 = *v46;
              if ( v26 != *v46 )
              {
                v61 = 1;
                v6 = 0;
                while ( v60 != -8 )
                {
                  if ( v60 == -16 && !v6 )
                    v6 = v46;
                  v59 = v58 & (v61 + v59);
                  v46 = &v10[v59];
                  v60 = *v46;
                  if ( v26 == *v46 )
                    goto LABEL_106;
                  ++v61;
                }
                if ( v6 )
                  v46 = v6;
                goto LABEL_106;
              }
              goto LABEL_75;
            }
LABEL_130:
            LODWORD(v73) = (2 * ((unsigned int)v73 >> 1) + 2) | v73 & 1;
            BUG();
          }
          sub_19628C0((__int64)&v72, 2 * v30);
          if ( (v73 & 1) != 0 )
          {
            v6 = (__int64 *)&v74;
            LODWORD(v10) = 7;
          }
          else
          {
            v6 = v74;
            if ( !v75 )
              goto LABEL_130;
            LODWORD(v10) = v75 - 1;
          }
          v54 = (unsigned int)v10 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v46 = &v6[v54];
          v45 = v73;
          v55 = *v46;
          if ( v26 != *v46 )
          {
            v56 = 1;
            v57 = 0;
            while ( v55 != -8 )
            {
              if ( v55 == -16 && !v57 )
                v57 = v46;
              v54 = (unsigned int)v10 & (v56 + v54);
              v46 = &v6[v54];
              v55 = *v46;
              if ( v26 == *v46 )
                goto LABEL_106;
              ++v56;
            }
            if ( v57 )
            {
              v46 = v57;
              v45 = v73;
            }
            else
            {
LABEL_106:
              v45 = v73;
            }
          }
LABEL_75:
          LODWORD(v73) = (2 * (v45 >> 1) + 2) | v45 & 1;
          if ( *v46 != -8 )
            --HIDWORD(v73);
          *v46 = v26;
          v49 = (unsigned int)v77;
          if ( (unsigned int)v77 >= HIDWORD(v77) )
          {
            sub_16CD150((__int64)&v76, v78, 0, 8, (int)v10, (int)v6);
            v49 = (unsigned int)v77;
          }
          *(_QWORD *)&v76[8 * v49] = v26;
          v14 = (unsigned int)v70;
          LODWORD(v77) = v77 + 1;
          if ( (unsigned int)v70 >= HIDWORD(v70) )
          {
            sub_16CD150((__int64)&v69, v71, 0, 8, (int)v10, (int)v6);
            v14 = (unsigned int)v70;
          }
          *(_QWORD *)&v69[8 * v14] = v26;
          v16 = v69;
          LODWORD(v14) = v70 + 1;
          LODWORD(v70) = v70 + 1;
          goto LABEL_13;
        }
      }
LABEL_28:
      LODWORD(v14) = v70;
      v16 = v69;
      v17 = &v69[8 * (unsigned int)v70];
      if ( !(_DWORD)v70 )
        goto LABEL_29;
    }
    v43 = *(_QWORD *)(v5 + 72);
    if ( v43 == *(_QWORD *)(v5 + 64) )
      v44 = *(unsigned int *)(v5 + 84);
    else
      v44 = *(unsigned int *)(v5 + 80);
    v53 = (_QWORD *)(v43 + 8 * v44);
    goto LABEL_64;
  }
LABEL_29:
  if ( v16 != v71 )
    _libc_free((unsigned __int64)v16);
  v34 = (unsigned __int64)v76;
  v69 = v71;
  v70 = 0x800000000LL;
  if ( v76 != &v76[8 * (unsigned int)v77] )
  {
    v62 = v5;
    v35 = &v76[8 * (unsigned int)v77];
    while ( 1 )
    {
      v36 = *(_QWORD *)v34;
      v37 = *(_QWORD *)(*(_QWORD *)v34 + 48LL);
      v38 = *(_QWORD *)v34 + 40LL;
      if ( v38 != v37 )
        break;
LABEL_46:
      v34 += 8LL;
      if ( v35 == (_BYTE *)v34 )
      {
        v5 = v62;
        goto LABEL_48;
      }
    }
    while ( 1 )
    {
      if ( !v37 )
        BUG();
      v40 = *(_QWORD *)(v37 - 16);
      if ( !v40 )
        goto LABEL_39;
      if ( *(_QWORD *)(v40 + 8) || (v41 = sub_1648700(v40), v36 != v41[5]) || *((_BYTE *)v41 + 16) == 77 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v37 - 24) + 8LL) != 10 )
        {
          v39 = (unsigned int)v70;
          if ( (unsigned int)v70 >= HIDWORD(v70) )
          {
            sub_16CD150((__int64)&v69, v71, 0, 8, (int)v10, (int)v6);
            v39 = (unsigned int)v70;
          }
          *(_QWORD *)&v69[8 * v39] = v37 - 24;
          LODWORD(v70) = v70 + 1;
        }
LABEL_39:
        v37 = *(_QWORD *)(v37 + 8);
        if ( v38 == v37 )
          goto LABEL_46;
      }
      else
      {
        v37 = *(_QWORD *)(v37 + 8);
        if ( v38 == v37 )
          goto LABEL_46;
      }
    }
  }
LABEL_48:
  v42 = sub_1AE2630((__int64)&v69, a2, a3);
  v4 = v42;
  if ( a4 && (_BYTE)v42 )
    sub_1465150(a4, v5);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( (v73 & 1) == 0 )
    j___libc_free_0(v74);
LABEL_2:
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return v4;
}
