// Function: sub_3383BC0
// Address: 0x3383bc0
//
__int64 __fastcall sub_3383BC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r8
  int v12; // eax
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rsi
  __int64 v17; // r12
  unsigned int v19; // esi
  __int64 v20; // r10
  _QWORD *v21; // rcx
  __int64 v22; // r9
  _DWORD *v23; // rcx
  unsigned int v24; // esi
  __int64 v25; // r15
  __int64 v26; // r10
  __int64 v27; // r8
  __int64 v28; // rdi
  __int64 v29; // rdx
  _QWORD *v30; // r15
  __int64 *v31; // rdi
  int v32; // ecx
  int v33; // r10d
  _QWORD *v34; // rdx
  int v35; // edi
  int v36; // edi
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // r9
  unsigned int v40; // esi
  __int64 v41; // r8
  int v42; // r11d
  _QWORD *v43; // r10
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // r9
  __int64 v47; // r11
  __int64 v48; // rsi
  __int64 *v49; // rdx
  int v50; // ecx
  int v51; // r11d
  __int64 v52; // r9
  unsigned int v53; // ecx
  __int64 v54; // rsi
  int v55; // esi
  int v56; // esi
  __int64 v57; // r9
  int v58; // r11d
  unsigned int v59; // ecx
  __int64 v60; // r8
  int v61; // [rsp+14h] [rbp-5Ch]
  int v62; // [rsp+14h] [rbp-5Ch]
  unsigned int v63; // [rsp+18h] [rbp-58h]
  int v64; // [rsp+18h] [rbp-58h]
  __int64 v65; // [rsp+18h] [rbp-58h]
  unsigned int v66; // [rsp+20h] [rbp-50h]
  int v67; // [rsp+20h] [rbp-50h]
  int v68; // [rsp+20h] [rbp-50h]
  __int64 v69; // [rsp+20h] [rbp-50h]
  int v70; // [rsp+20h] [rbp-50h]
  int v71; // [rsp+20h] [rbp-50h]
  unsigned int v72; // [rsp+20h] [rbp-50h]
  __int64 v74; // [rsp+28h] [rbp-48h]
  unsigned int v75; // [rsp+28h] [rbp-48h]
  __int64 v76; // [rsp+28h] [rbp-48h]
  int v77; // [rsp+28h] [rbp-48h]
  int v78; // [rsp+28h] [rbp-48h]
  __int64 v79; // [rsp+30h] [rbp-40h] BYREF
  int v80; // [rsp+38h] [rbp-38h]

  v9 = *(_QWORD *)(a1[108] + 40);
  v10 = sub_E6C430(*(_QWORD *)(v9 + 24), a2, a3, a4, (__int64)a5);
  *a5 = v10;
  v11 = v10;
  v12 = *(_DWORD *)(a1[120] + 892);
  if ( v12 )
  {
    v19 = *(_DWORD *)(v9 + 544);
    if ( v19 )
    {
      v20 = *(_QWORD *)(v9 + 528);
      v66 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
      v63 = (v19 - 1) & v66;
      v21 = (_QWORD *)(v20 + 16LL * v63);
      v22 = *v21;
      if ( v11 == *v21 )
      {
LABEL_11:
        v23 = v21 + 1;
        goto LABEL_12;
      }
      v61 = 1;
      v31 = 0;
      while ( v22 != -4096 )
      {
        if ( !v31 && v22 == -8192 )
          v31 = v21;
        v63 = (v19 - 1) & (v61 + v63);
        v21 = (_QWORD *)(v20 + 16LL * v63);
        v22 = *v21;
        if ( v11 == *v21 )
          goto LABEL_11;
        ++v61;
      }
      if ( !v31 )
        v31 = v21;
      v32 = *(_DWORD *)(v9 + 536);
      ++*(_QWORD *)(v9 + 520);
      v33 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v19 )
      {
        if ( v19 - *(_DWORD *)(v9 + 540) - v33 > v19 >> 3 )
        {
LABEL_23:
          *(_DWORD *)(v9 + 536) = v33;
          if ( *v31 != -4096 )
            --*(_DWORD *)(v9 + 540);
          *v31 = v11;
          v23 = v31 + 1;
          *((_DWORD *)v31 + 2) = 0;
LABEL_12:
          *v23 = v12;
          v24 = *((_DWORD *)a1 + 252);
          v25 = *(_QWORD *)(*(_QWORD *)(a1[120] + 56) + 8LL * *(unsigned int *)(a4 + 44));
          if ( v24 )
          {
            v26 = a1[124];
            v75 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v27 = v26 + 40LL * v75;
            v28 = *(_QWORD *)v27;
            if ( v25 == *(_QWORD *)v27 )
            {
LABEL_14:
              v29 = *(unsigned int *)(v27 + 16);
              v30 = (_QWORD *)(v27 + 8);
              if ( *(unsigned int *)(v27 + 20) < (unsigned __int64)(v29 + 1) )
              {
                v67 = v12;
                v76 = v27;
                sub_C8D5F0(v27 + 8, (const void *)(v27 + 24), v29 + 1, 4u, v27, v29 + 1);
                v12 = v67;
                v29 = *(unsigned int *)(v76 + 16);
              }
LABEL_16:
              *(_DWORD *)(*v30 + 4 * v29) = v12;
              ++*((_DWORD *)v30 + 2);
              *(_DWORD *)(a1[120] + 892) = 0;
              v11 = *a5;
              goto LABEL_2;
            }
            v68 = 1;
            v34 = 0;
            while ( v28 != -4096 )
            {
              if ( v28 == -8192 && !v34 )
                v34 = (_QWORD *)v27;
              v75 = (v24 - 1) & (v68 + v75);
              v27 = v26 + 40LL * v75;
              v28 = *(_QWORD *)v27;
              if ( v25 == *(_QWORD *)v27 )
                goto LABEL_14;
              ++v68;
            }
            v35 = *((_DWORD *)a1 + 250);
            if ( !v34 )
              v34 = (_QWORD *)v27;
            ++a1[123];
            v36 = v35 + 1;
            if ( 4 * v36 < 3 * v24 )
            {
              if ( v24 - *((_DWORD *)a1 + 251) - v36 > v24 >> 3 )
              {
LABEL_32:
                *((_DWORD *)a1 + 250) = v36;
                if ( *v34 != -4096 )
                  --*((_DWORD *)a1 + 251);
                *v34 = v25;
                v30 = v34 + 1;
                v34[1] = v34 + 3;
                v34[2] = 0x400000000LL;
                v29 = 0;
                goto LABEL_16;
              }
              v72 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
              v78 = v12;
              sub_3383980((__int64)(a1 + 123), v24);
              v55 = *((_DWORD *)a1 + 252);
              if ( v55 )
              {
                v56 = v55 - 1;
                v57 = a1[124];
                v43 = 0;
                v58 = 1;
                v59 = v56 & v72;
                v36 = *((_DWORD *)a1 + 250) + 1;
                v12 = v78;
                v34 = (_QWORD *)(v57 + 40LL * (v56 & v72));
                v60 = *v34;
                if ( v25 == *v34 )
                  goto LABEL_32;
                while ( v60 != -4096 )
                {
                  if ( v60 == -8192 && !v43 )
                    v43 = v34;
                  v59 = v56 & (v58 + v59);
                  v34 = (_QWORD *)(v57 + 40LL * v59);
                  v60 = *v34;
                  if ( v25 == *v34 )
                    goto LABEL_32;
                  ++v58;
                }
                goto LABEL_40;
              }
              goto LABEL_88;
            }
          }
          else
          {
            ++a1[123];
          }
          v77 = v12;
          sub_3383980((__int64)(a1 + 123), 2 * v24);
          v37 = *((_DWORD *)a1 + 252);
          if ( v37 )
          {
            v38 = v37 - 1;
            v39 = a1[124];
            v40 = v38 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v36 = *((_DWORD *)a1 + 250) + 1;
            v12 = v77;
            v34 = (_QWORD *)(v39 + 40LL * v40);
            v41 = *v34;
            if ( v25 == *v34 )
              goto LABEL_32;
            v42 = 1;
            v43 = 0;
            while ( v41 != -4096 )
            {
              if ( !v43 && v41 == -8192 )
                v43 = v34;
              v40 = v38 & (v42 + v40);
              v34 = (_QWORD *)(v39 + 40LL * v40);
              v41 = *v34;
              if ( v25 == *v34 )
                goto LABEL_32;
              ++v42;
            }
LABEL_40:
            if ( v43 )
              v34 = v43;
            goto LABEL_32;
          }
LABEL_88:
          ++*((_DWORD *)a1 + 250);
          BUG();
        }
        v62 = v12;
        v65 = v11;
        sub_E9EE70(v9 + 520, v19);
        v50 = *(_DWORD *)(v9 + 544);
        if ( v50 )
        {
          v51 = v50 - 1;
          v52 = *(_QWORD *)(v9 + 528);
          v11 = v65;
          v53 = v51 & v66;
          v33 = *(_DWORD *)(v9 + 536) + 1;
          v12 = v62;
          v31 = (__int64 *)(v52 + 16LL * (v51 & v66));
          v54 = *v31;
          if ( v65 == *v31 )
            goto LABEL_23;
          v71 = 1;
          v49 = 0;
          while ( v54 != -4096 )
          {
            if ( !v49 && v54 == -8192 )
              v49 = v31;
            v53 = v51 & (v71 + v53);
            v31 = (__int64 *)(v52 + 16LL * v53);
            v54 = *v31;
            if ( v65 == *v31 )
              goto LABEL_23;
            ++v71;
          }
          goto LABEL_48;
        }
        goto LABEL_89;
      }
    }
    else
    {
      ++*(_QWORD *)(v9 + 520);
    }
    v64 = v12;
    v69 = v11;
    sub_E9EE70(v9 + 520, 2 * v19);
    v44 = *(_DWORD *)(v9 + 544);
    if ( v44 )
    {
      v11 = v69;
      v45 = v44 - 1;
      v46 = *(_QWORD *)(v9 + 528);
      v33 = *(_DWORD *)(v9 + 536) + 1;
      v12 = v64;
      v47 = v45 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
      v31 = (__int64 *)(v46 + 16 * v47);
      v48 = *v31;
      if ( v69 == *v31 )
        goto LABEL_23;
      v70 = 1;
      v49 = 0;
      while ( v48 != -4096 )
      {
        if ( !v49 && v48 == -8192 )
          v49 = v31;
        LODWORD(v47) = v45 & (v70 + v47);
        v31 = (__int64 *)(v46 + 16LL * (unsigned int)v47);
        v48 = *v31;
        if ( v11 == *v31 )
          goto LABEL_23;
        ++v70;
      }
LABEL_48:
      if ( v49 )
        v31 = v49;
      goto LABEL_23;
    }
LABEL_89:
    ++*(_DWORD *)(v9 + 536);
    BUG();
  }
LABEL_2:
  v13 = *((_DWORD *)a1 + 212);
  v14 = *a1;
  v79 = 0;
  v15 = a1[108];
  v80 = v13;
  if ( v14 )
  {
    if ( &v79 != (__int64 *)(v14 + 48) )
    {
      v16 = *(_QWORD *)(v14 + 48);
      v79 = v16;
      if ( v16 )
      {
        v74 = v11;
        sub_B96E90((__int64)&v79, v16, 1);
        v11 = v74;
      }
    }
  }
  v17 = sub_33F2D10(v15, &v79, a2, a3, v11);
  if ( v79 )
    sub_B91220((__int64)&v79, v79);
  return v17;
}
