// Function: sub_27E5160
// Address: 0x27e5160
//
__int64 __fastcall sub_27E5160(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rax
  unsigned int v5; // ebx
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int8 *v11; // rsi
  unsigned __int8 *v12; // rsi
  unsigned __int8 **v13; // r13
  unsigned int v14; // ebx
  unsigned int v15; // r12d
  unsigned __int8 **v16; // r14
  unsigned __int8 *v17; // rdi
  unsigned int v18; // r15d
  bool v19; // al
  unsigned int v20; // ecx
  unsigned __int8 **v21; // r8
  unsigned __int64 v22; // rdx
  __int64 *v23; // r11
  __int64 v24; // r9
  unsigned __int8 **v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int8 **v28; // r13
  __int64 v29; // r8
  unsigned __int8 *v30; // r12
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  __int64 *v33; // r8
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 *v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rsi
  unsigned __int64 v40; // rcx
  __int64 v41; // rsi
  unsigned __int64 v42; // rcx
  __int64 v43; // rsi
  unsigned __int64 v44; // rcx
  unsigned int v45; // eax
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rax
  unsigned int v49; // ebx
  int v50; // eax
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rax
  __int64 v59; // [rsp+10h] [rbp-140h]
  __int64 v60; // [rsp+10h] [rbp-140h]
  __int64 v62; // [rsp+18h] [rbp-138h]
  unsigned __int64 v63; // [rsp+20h] [rbp-130h]
  __int64 v64; // [rsp+20h] [rbp-130h]
  __int64 *v65; // [rsp+20h] [rbp-130h]
  __int64 v66; // [rsp+28h] [rbp-128h]
  __int64 *v67; // [rsp+28h] [rbp-128h]
  __int64 v68; // [rsp+28h] [rbp-128h]
  unsigned __int8 v69; // [rsp+38h] [rbp-118h]
  __int64 *v70; // [rsp+38h] [rbp-118h]
  __int64 *v71; // [rsp+38h] [rbp-118h]
  __int64 *v72; // [rsp+40h] [rbp-110h] BYREF
  unsigned __int64 v73; // [rsp+48h] [rbp-108h]
  __int64 v74; // [rsp+50h] [rbp-100h] BYREF
  int v75; // [rsp+58h] [rbp-F8h]
  char v76; // [rsp+5Ch] [rbp-F4h]
  _BYTE v77[48]; // [rsp+60h] [rbp-F0h] BYREF
  unsigned __int8 **v78; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+98h] [rbp-B8h]
  _BYTE v80[176]; // [rsp+A0h] [rbp-B0h] BYREF

  if ( **(_BYTE **)(a2 - 64) == 17 )
    return 0;
  v2 = a2;
  if ( **(_BYTE **)(a2 - 32) == 17 )
    return 0;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 56);
  if ( !v4 )
    BUG();
  v5 = 0;
  if ( *(_BYTE *)(v4 - 24) == 84 )
  {
    v7 = a1;
    v8 = sub_AA4FF0(*(_QWORD *)(a2 + 40));
    if ( !v8 )
      BUG();
    v9 = (unsigned int)*(unsigned __int8 *)(v8 - 24) - 39;
    if ( (unsigned int)v9 > 0x38 || (v10 = 0x100060000000001LL, !_bittest64(&v10, v9)) )
    {
      v11 = *(unsigned __int8 **)(a2 - 64);
      v78 = (unsigned __int8 **)v80;
      v79 = 0x800000000LL;
      v72 = 0;
      v73 = (unsigned __int64)v77;
      v74 = 4;
      v75 = 0;
      v76 = 1;
      v69 = sub_27DEC50(a1, v11, v3, (__int64)&v78, 0, (__int64)&v72, (unsigned __int8 *)v2);
      if ( !v76 )
        _libc_free(v73);
      if ( !v69 )
      {
        v12 = *(unsigned __int8 **)(v2 - 32);
        v73 = (unsigned __int64)v77;
        v72 = 0;
        v74 = 4;
        v75 = 0;
        v76 = 1;
        v5 = sub_27DEC50(a1, v12, v3, (__int64)&v78, 0, (__int64)&v72, (unsigned __int8 *)v2);
        if ( !v76 )
          _libc_free(v73);
        if ( !(_BYTE)v5 )
          goto LABEL_62;
      }
      if ( &v78[2 * (unsigned int)v79] == v78 )
      {
        v23 = &v74;
        v72 = &v74;
        v73 = 0x800000000LL;
        v55 = *(_QWORD *)(v3 + 56);
        if ( v55 )
        {
          if ( (*(_DWORD *)(v55 - 20) & 0x7FFFFFF) != 0 )
          {
            v36 = &v74;
            v33 = &v74;
            v34 = 0;
            goto LABEL_90;
          }
          goto LABEL_110;
        }
LABEL_114:
        BUG();
      }
      v13 = &v78[2 * (unsigned int)v79];
      v59 = v2;
      v14 = 0;
      v15 = 0;
      v16 = v78;
      v66 = v3;
      v63 = (unsigned __int64)v78;
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = *v16;
          if ( (unsigned int)**v16 - 12 > 1 )
            break;
LABEL_17:
          v16 += 2;
          if ( v13 == v16 )
            goto LABEL_23;
        }
        v18 = *((_DWORD *)v17 + 8);
        if ( v18 <= 0x40 )
          v19 = *((_QWORD *)v17 + 3) == 0;
        else
          v19 = v18 == (unsigned int)sub_C444A0((__int64)(v17 + 24));
        if ( !v19 )
        {
          ++v14;
          goto LABEL_17;
        }
        v16 += 2;
        ++v15;
        if ( v13 == v16 )
        {
LABEL_23:
          v20 = v15;
          v21 = v13;
          v3 = v66;
          v22 = v63;
          v7 = a1;
          v2 = v59;
          if ( v14 > v20 )
          {
            v48 = (__int64 *)sub_AA48A0(v66);
            v47 = sub_ACD6D0(v48);
          }
          else
          {
            if ( !(v20 | v14) )
            {
              v23 = &v74;
              v24 = 0;
              v72 = &v74;
              v73 = 0x800000000LL;
LABEL_26:
              v25 = (unsigned __int8 **)v22;
              v26 = a1;
              v27 = 0;
              v28 = v21;
              v29 = v59;
              do
              {
                if ( *v25 == (unsigned __int8 *)v24 || (unsigned int)**v25 - 12 <= 1 )
                {
                  v30 = v25[1];
                  if ( v27 + 1 > (unsigned __int64)HIDWORD(v73) )
                  {
                    v60 = v29;
                    v62 = v26;
                    v64 = v24;
                    v67 = v23;
                    sub_C8D5F0((__int64)&v72, v23, v27 + 1, 8u, v29, v24);
                    v27 = (unsigned int)v73;
                    v29 = v60;
                    v26 = v62;
                    v24 = v64;
                    v23 = v67;
                  }
                  v72[v27] = (__int64)v30;
                  v27 = (unsigned int)(v73 + 1);
                  LODWORD(v73) = v73 + 1;
                }
                v25 += 2;
              }
              while ( v28 != v25 );
              v7 = v26;
              v2 = v29;
              goto LABEL_34;
            }
            v46 = (__int64 *)sub_AA48A0(v66);
            v47 = sub_ACD720(v46);
          }
          v22 = (unsigned __int64)v78;
          v24 = v47;
          v23 = &v74;
          v72 = &v74;
          v73 = 0x800000000LL;
          v21 = &v78[2 * (unsigned int)v79];
          if ( v78 != v21 )
            goto LABEL_26;
          v27 = 0;
LABEL_34:
          v31 = *(_QWORD *)(v3 + 56);
          if ( v31 )
          {
            if ( (*(_DWORD *)(v31 - 20) & 0x7FFFFFF) != v27 )
            {
              v32 = (unsigned __int64)v72;
              v33 = &v72[v27];
              v34 = (8 * v27) >> 3;
              v35 = (8 * v27) >> 5;
              if ( v35 )
              {
                v36 = v72;
                v37 = &v72[4 * v35];
                while ( 1 )
                {
                  v38 = *(_QWORD *)(*v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v38 == *v36 + 48 )
                    goto LABEL_112;
                  if ( !v38 )
                    BUG();
                  if ( (unsigned int)*(unsigned __int8 *)(v38 - 24) - 30 > 0xA )
LABEL_112:
                    BUG();
                  if ( *(_BYTE *)(v38 - 24) == 33 )
                    break;
                  v39 = v36[1];
                  v40 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v40 == v39 + 48 )
                    goto LABEL_122;
                  if ( !v40 )
                    BUG();
                  if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 > 0xA )
LABEL_122:
                    BUG();
                  if ( *(_BYTE *)(v40 - 24) == 33 )
                  {
                    ++v36;
                    break;
                  }
                  v41 = v36[2];
                  v42 = *(_QWORD *)(v41 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v42 == v41 + 48 )
                    goto LABEL_117;
                  if ( !v42 )
                    BUG();
                  if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
LABEL_117:
                    BUG();
                  if ( *(_BYTE *)(v42 - 24) == 33 )
                  {
                    v36 += 2;
                    break;
                  }
                  v43 = v36[3];
                  v44 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v44 == v43 + 48 )
                    goto LABEL_126;
                  if ( !v44 )
                    BUG();
                  if ( (unsigned int)*(unsigned __int8 *)(v44 - 24) - 30 > 0xA )
LABEL_126:
                    BUG();
                  if ( *(_BYTE *)(v44 - 24) == 33 )
                  {
                    v36 += 3;
                    break;
                  }
                  v36 += 4;
                  if ( v37 == v36 )
                  {
                    v34 = v33 - v36;
                    goto LABEL_56;
                  }
                }
LABEL_71:
                v5 = 0;
                if ( v36 == v33 )
                {
LABEL_59:
                  v70 = v23;
                  v45 = sub_27E4FC0(v7, v3, (__int64)&v72);
                  v32 = (unsigned __int64)v72;
                  v23 = v70;
                  v5 = v45;
                }
LABEL_60:
                if ( (__int64 *)v32 != v23 )
                  _libc_free(v32);
LABEL_62:
                if ( v78 != (unsigned __int8 **)v80 )
                  _libc_free((unsigned __int64)v78);
                return v5;
              }
              v36 = v72;
LABEL_90:
              v32 = (unsigned __int64)v36;
LABEL_56:
              if ( v34 != 2 )
              {
                if ( v34 != 3 )
                {
                  if ( v34 != 1 )
                    goto LABEL_59;
                  goto LABEL_101;
                }
                v56 = *(_QWORD *)(*v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v56 == *v36 + 48 )
                  goto LABEL_120;
                if ( !v56 )
                  BUG();
                if ( (unsigned int)*(unsigned __int8 *)(v56 - 24) - 30 > 0xA )
LABEL_120:
                  BUG();
                if ( *(_BYTE *)(v56 - 24) == 33 )
                  goto LABEL_71;
                ++v36;
              }
              v57 = *(_QWORD *)(*v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v57 == *v36 + 48 )
                goto LABEL_116;
              if ( !v57 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v57 - 24) - 30 > 0xA )
LABEL_116:
                BUG();
              if ( *(_BYTE *)(v57 - 24) == 33 )
                goto LABEL_71;
              ++v36;
LABEL_101:
              v58 = *(_QWORD *)(*v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v58 == *v36 + 48 )
                goto LABEL_128;
              if ( !v58 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v58 - 24) - 30 > 0xA )
LABEL_128:
                BUG();
              if ( *(_BYTE *)(v58 - 24) == 33 )
                goto LABEL_71;
              goto LABEL_59;
            }
            if ( v24 )
            {
              v49 = *(_DWORD *)(v24 + 32);
              if ( v49 <= 0x40 )
              {
                if ( !*(_QWORD *)(v24 + 24) )
                  goto LABEL_77;
              }
              else
              {
                v65 = v23;
                v68 = v24;
                v50 = sub_C444A0(v24 + 24);
                v24 = v68;
                v23 = v65;
                if ( v49 == v50 )
                {
LABEL_77:
                  v51 = *(_QWORD *)(v2 + 32LL * v69 - 64);
                  if ( v51 && v2 == v51 )
                    goto LABEL_79;
                  v71 = v23;
LABEL_107:
                  v5 = 1;
                  sub_BD84D0(v2, v51);
                  sub_B43D60((_QWORD *)v2);
                  v32 = (unsigned __int64)v72;
                  v23 = v71;
                  goto LABEL_60;
                }
              }
LABEL_79:
              v52 = v2 + 32LL * (v69 ^ 1u) - 64;
              if ( *(_QWORD *)v52 )
              {
                v53 = *(_QWORD *)(v52 + 8);
                **(_QWORD **)(v52 + 16) = v53;
                if ( v53 )
                  *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
              }
              *(_QWORD *)v52 = v24;
              v54 = *(_QWORD *)(v24 + 16);
              *(_QWORD *)(v52 + 8) = v54;
              if ( v54 )
                *(_QWORD *)(v54 + 16) = v52 + 8;
              *(_QWORD *)(v52 + 16) = v24 + 16;
              v5 = 1;
              *(_QWORD *)(v24 + 16) = v52;
              v32 = (unsigned __int64)v72;
              goto LABEL_60;
            }
LABEL_110:
            v71 = v23;
            v51 = sub_ACA8A0(*(__int64 ***)(v2 + 8));
            goto LABEL_107;
          }
          goto LABEL_114;
        }
      }
    }
  }
  return v5;
}
