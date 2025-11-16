// Function: sub_2E12D10
// Address: 0x2e12d10
//
void __fastcall sub_2E12D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned int v12; // edx
  __int64 v13; // rcx
  __int64 *v14; // rbx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  unsigned int v19; // r12d
  __int64 *v20; // rbx
  __int64 v21; // r15
  __int64 v22; // r14
  __int64 v23; // r14
  __int64 v24; // r9
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 *v27; // rax
  int v28; // eax
  _QWORD *v29; // rbx
  __int64 v30; // r14
  unsigned int v31; // r15d
  __int64 v32; // rax
  __int64 v33; // r12
  unsigned __int64 v34; // rsi
  _BYTE *v35; // r8
  __int64 *v36; // rdi
  __int64 v37; // r9
  __int64 v38; // r10
  __int64 v39; // rax
  unsigned int v40; // ecx
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 v43; // r10
  __int64 v44; // r11
  __int64 v45; // rcx
  char v46; // r10
  __int64 v47; // r11
  __int64 v48; // r9
  unsigned __int16 v49; // ax
  __int64 *v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  _QWORD *v53; // rax
  __int64 v54; // r15
  unsigned __int64 v55; // r14
  _QWORD *v56; // rdx
  _QWORD *v57; // rsi
  __int64 v58; // rax
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // [rsp+8h] [rbp-118h]
  __int64 v64; // [rsp+10h] [rbp-110h]
  __int64 v65; // [rsp+18h] [rbp-108h]
  __int64 v66; // [rsp+28h] [rbp-F8h]
  __int64 v67; // [rsp+30h] [rbp-F0h]
  char v69; // [rsp+43h] [rbp-DDh]
  unsigned __int8 v70; // [rsp+43h] [rbp-DDh]
  int v71; // [rsp+44h] [rbp-DCh]
  __int16 *v72; // [rsp+48h] [rbp-D8h]
  __int64 v73; // [rsp+48h] [rbp-D8h]
  __int64 v74; // [rsp+50h] [rbp-D0h]
  char v75; // [rsp+58h] [rbp-C8h]
  int v76; // [rsp+5Ch] [rbp-C4h]
  _BYTE *v77; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+68h] [rbp-B8h]
  _BYTE v79[176]; // [rsp+70h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  v77 = v79;
  v78 = 0x800000000LL;
  v71 = *(_DWORD *)(v5 + 64);
  v76 = 0;
  if ( v71 )
  {
    while ( 1 )
    {
      v8 = v76 & 0x7FFFFFFF;
      v9 = v76 & 0x7FFFFFFF;
      v10 = *(_QWORD *)(*(_QWORD *)(v5 + 56) + 16 * v9 + 8);
      if ( !v10 )
        goto LABEL_6;
      while ( (*(_BYTE *)(v10 + 4) & 8) != 0 )
      {
        v10 = *(_QWORD *)(v10 + 32);
        if ( !v10 )
          goto LABEL_6;
      }
      v11 = *(unsigned int *)(a1 + 160);
      if ( v8 >= (unsigned int)v11 )
        break;
      v74 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * v9);
      if ( !v74 )
        break;
LABEL_4:
      if ( *(_DWORD *)(v74 + 8) )
      {
        v7 = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 4 * v9);
        if ( (_DWORD)v7 )
        {
          v16 = *(_QWORD **)(a1 + 16);
          LODWORD(v78) = 0;
          v17 = v16[7];
          v18 = v16[1] + 24 * v7;
          v19 = *(_DWORD *)(v18 + 16) & 0xFFF;
          v20 = (__int64 *)(v16[8] + 16LL * *(unsigned __int16 *)(v18 + 20));
          if ( v17 + 2LL * (*(_DWORD *)(v18 + 16) >> 12) )
          {
            v72 = (__int16 *)(v17 + 2LL * (*(_DWORD *)(v18 + 16) >> 12));
            v66 = 0;
            v67 = 0;
            while ( 1 )
            {
              v21 = *v20;
              v22 = v20[1];
              if ( (unsigned __int8)sub_E92100((__int64)v16, v19) )
              {
                v67 |= v21;
                v66 |= v22;
              }
              v23 = *(_QWORD *)(*(_QWORD *)(a1 + 424) + 8LL * v19);
              if ( !v23 )
              {
                v69 = qword_501EA48[8];
                v53 = (_QWORD *)sub_22077B0(0x68u);
                v23 = (__int64)v53;
                if ( v53 )
                {
                  *v53 = v53 + 2;
                  v53[1] = 0x200000000LL;
                  v53[8] = v53 + 10;
                  v53[9] = 0x200000000LL;
                  if ( v69 )
                  {
                    v58 = sub_22077B0(0x30u);
                    if ( v58 )
                    {
                      *(_DWORD *)(v58 + 8) = 0;
                      *(_QWORD *)(v58 + 16) = 0;
                      *(_QWORD *)(v58 + 24) = v58 + 8;
                      *(_QWORD *)(v58 + 32) = v58 + 8;
                      *(_QWORD *)(v58 + 40) = 0;
                    }
                    *(_QWORD *)(v23 + 96) = v58;
                  }
                  else
                  {
                    v53[12] = 0;
                  }
                }
                *(_QWORD *)(*(_QWORD *)(a1 + 424) + 8LL * v19) = v23;
                sub_2E11710((_QWORD *)a1, v23, v19);
              }
              if ( *(_DWORD *)(v23 + 8) )
              {
                v25 = sub_2E09D00((__int64 *)v23, *(_QWORD *)(*(_QWORD *)v74 + 8LL));
                v26 = (unsigned int)v78;
                v18 = (unsigned int)v78 + 1LL;
                if ( v18 > HIDWORD(v78) )
                {
                  sub_C8D5F0((__int64)&v77, v79, v18, 0x10u, a5, v24);
                  v26 = (unsigned int)v78;
                }
                v27 = (__int64 *)&v77[16 * v26];
                *v27 = v23;
                v27[1] = v25;
                LODWORD(v78) = v78 + 1;
              }
              v20 += 2;
              v28 = *v72++;
              if ( !(_WORD)v28 )
                break;
              v16 = *(_QWORD **)(a1 + 16);
              v19 += v28;
            }
          }
          else
          {
            v66 = 0;
            v67 = 0;
          }
          v29 = *(_QWORD **)v74;
          v30 = *(_QWORD *)v74 + 24LL * *(unsigned int *)(v74 + 8);
          if ( v30 != *(_QWORD *)v74 )
          {
            v31 = v76 | 0x80000000;
            while ( 1 )
            {
              v32 = v29[1];
              v29 += 3;
              if ( (v32 & 6) != 0 )
              {
                v33 = *(_QWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 16);
                if ( v33 )
                  break;
              }
LABEL_42:
              if ( (_QWORD *)v30 == v29 )
                goto LABEL_6;
            }
            v34 = (unsigned __int64)v77;
            v35 = &v77[16 * (unsigned int)v78];
            if ( v35 != v77 )
            {
              do
              {
                v36 = *(__int64 **)v34;
                v37 = *(_QWORD *)(v34 + 8);
                v38 = **(_QWORD **)v34;
                v39 = 24LL * *(unsigned int *)(*(_QWORD *)v34 + 8LL);
                v18 = v38 + v39;
                if ( v37 != v38 + v39 )
                {
                  v40 = *(_DWORD *)((*(v29 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)*(v29 - 2) >> 1) & 3;
                  if ( v40 < (*(_DWORD *)((*(_QWORD *)(v38 + v39 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                            | (unsigned int)(*(__int64 *)(v38 + v39 - 16) >> 1) & 3) )
                  {
                    v18 = *(_QWORD *)(v34 + 8);
                    if ( v40 >= (*(_DWORD *)((*(_QWORD *)(v37 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                               | (unsigned int)(*(__int64 *)(v37 + 8) >> 1) & 3) )
                    {
                      do
                      {
                        v41 = *(_QWORD *)(v18 + 32);
                        v18 += 24LL;
                      }
                      while ( v40 >= (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v41 >> 1) & 3) );
                    }
                  }
                  *(_QWORD *)(v34 + 8) = v18;
                  if ( v18 != *v36 + 24LL * *((unsigned int *)v36 + 2) )
                  {
                    v18 = *(_DWORD *)((*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                        | (unsigned int)(*(__int64 *)v18 >> 1) & 3;
                    if ( (unsigned int)v18 < (*(_DWORD *)((*(v29 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                            | (unsigned int)((__int64)*(v29 - 2) >> 1) & 3) )
                      goto LABEL_41;
                  }
                }
                v34 += 16LL;
              }
              while ( v35 != (_BYTE *)v34 );
            }
            v42 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 48LL);
            if ( !(_BYTE)v42 )
              goto LABEL_63;
            v43 = *(_QWORD *)(v74 + 104);
            if ( v43 )
            {
              v44 = v66;
              v59 = v67;
              do
              {
                v18 = *(_QWORD *)v43;
                v60 = *(_QWORD *)v43 + 24LL * *(unsigned int *)(v43 + 8);
                if ( v60 != *(_QWORD *)v43 )
                {
                  v61 = *(v29 - 2);
                  while ( 1 )
                  {
                    if ( (*(_DWORD *)((*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                        | (unsigned int)(*(__int64 *)v18 >> 1) & 3) >= (*(_DWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                      | (unsigned int)((__int64)*(v29 - 2) >> 1) & 3) )
                      goto LABEL_87;
                    if ( *(_QWORD *)(v18 + 8) == v61 )
                      break;
                    v18 += 24LL;
                    if ( v60 == v18 )
                      goto LABEL_87;
                  }
                  v59 |= *(_QWORD *)(v43 + 112);
                  v44 |= *(_QWORD *)(v43 + 120);
                }
LABEL_87:
                v43 = *(_QWORD *)(v43 + 104);
              }
              while ( v43 );
              v73 = v59;
              v42 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 48LL);
            }
            else
            {
              v73 = -1;
              v44 = -1;
            }
            v45 = *(_QWORD *)(v33 + 32);
            v46 = 0;
            v47 = ~v44;
            v48 = v45 + 40LL * (*(_DWORD *)(v33 + 40) & 0xFFFFFF);
            if ( v45 == v48 )
              goto LABEL_61;
            while ( 1 )
            {
LABEL_53:
              if ( *(_BYTE *)v45 || v31 != *(_DWORD *)(v45 + 8) )
                goto LABEL_52;
              v49 = (*(_DWORD *)v45 >> 8) & 0xFFF;
              if ( (*(_BYTE *)(v45 + 3) & 0x10) != 0 )
                break;
              if ( v49 )
              {
                v50 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL) + 16LL * v49);
                v51 = *v50;
                v52 = v50[1];
              }
              else
              {
                v63 = v47;
                v64 = v45;
                v65 = v48;
                v70 = v42;
                v75 = v46;
                v62 = sub_2EBF1E0(*(_QWORD *)(a1 + 8), v31, v18, v45, v42, v48);
                v47 = v63;
                v45 = v64;
                v48 = v65;
                v42 = v70;
                v51 = v62;
                v46 = v75;
              }
              v18 = v51 & ~v73 | v47 & v52;
              if ( v18 )
                goto LABEL_41;
              v45 += 40;
              if ( v48 == v45 )
              {
LABEL_60:
                if ( v46 )
                  goto LABEL_63;
LABEL_61:
                if ( v29 == (_QWORD *)(*(_QWORD *)v74 + 24LL * *(unsigned int *)(v74 + 8)) || *v29 != *(v29 - 2) )
                {
LABEL_63:
                  sub_2E8F280(v33, v31, 0, 0);
                  goto LABEL_42;
                }
LABEL_41:
                sub_2E8D6E0(v33, v31, 0);
                goto LABEL_42;
              }
            }
            if ( !v49 )
              v46 = v42;
LABEL_52:
            v45 += 40;
            if ( v48 == v45 )
              goto LABEL_60;
            goto LABEL_53;
          }
        }
      }
LABEL_6:
      if ( v71 == ++v76 )
      {
        if ( v77 != v79 )
          _libc_free((unsigned __int64)v77);
        return;
      }
      v5 = *(_QWORD *)(a1 + 8);
    }
    v12 = v8 + 1;
    if ( (unsigned int)v11 < v12 && v12 != v11 )
    {
      if ( v12 >= v11 )
      {
        v54 = *(_QWORD *)(a1 + 168);
        v55 = v12 - v11;
        if ( v12 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
        {
          sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v12, 8u, a5, v12);
          v11 = *(unsigned int *)(a1 + 160);
        }
        v13 = *(_QWORD *)(a1 + 152);
        v56 = (_QWORD *)(v13 + 8 * v11);
        v57 = &v56[v55];
        if ( v56 != v57 )
        {
          do
            *v56++ = v54;
          while ( v57 != v56 );
          LODWORD(v11) = *(_DWORD *)(a1 + 160);
          v13 = *(_QWORD *)(a1 + 152);
        }
        *(_DWORD *)(a1 + 160) = v55 + v11;
        goto LABEL_13;
      }
      *(_DWORD *)(a1 + 160) = v12;
    }
    v13 = *(_QWORD *)(a1 + 152);
LABEL_13:
    v14 = (__int64 *)(v13 + 8 * v9);
    v15 = sub_2E10F30(v76 | 0x80000000);
    *v14 = v15;
    v74 = v15;
    sub_2E11E80((_QWORD *)a1, v15);
    goto LABEL_4;
  }
}
