// Function: sub_37DD9E0
// Address: 0x37dd9e0
//
void __fastcall sub_37DD9E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 *v8; // r12
  unsigned int v9; // r13d
  __int64 *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r12
  unsigned __int64 v15; // r13
  unsigned int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // rcx
  _BYTE *v19; // r13
  int v20; // ebx
  int *v21; // rdi
  int v22; // r11d
  int *v23; // rdx
  _DWORD *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned __int64 v27; // rcx
  unsigned int v28; // esi
  int v29; // eax
  int v30; // edx
  int v31; // edx
  __int64 v32; // rcx
  int v33; // esi
  int v34; // ecx
  __int64 v35; // r13
  int v36; // r12d
  unsigned int v37; // eax
  __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  unsigned int v43; // esi
  int v44; // edx
  int v45; // r11d
  int *v46; // rax
  int v47; // edi
  unsigned int *v48; // r13
  __int64 v49; // rax
  __int64 v50; // rbx
  unsigned int *v51; // r12
  __m128i *v52; // rsi
  unsigned __int64 v53; // rax
  __int64 *v54; // rbx
  __int64 *v55; // rdi
  int v56; // edi
  int v57; // esi
  int v58; // ecx
  int v59; // edi
  int v60; // r11d
  int *v61; // rsi
  __int64 v62; // [rsp+0h] [rbp-E0h]
  __int64 v63; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v65; // [rsp+30h] [rbp-B0h]
  __int64 v66; // [rsp+30h] [rbp-B0h]
  __int64 *v67; // [rsp+38h] [rbp-A8h]
  int v68; // [rsp+4Ch] [rbp-94h] BYREF
  __int64 v69; // [rsp+50h] [rbp-90h] BYREF
  int *v70; // [rsp+58h] [rbp-88h] BYREF
  _BYTE *v71; // [rsp+60h] [rbp-80h] BYREF
  __int64 v72; // [rsp+68h] [rbp-78h]
  _BYTE v73[112]; // [rsp+70h] [rbp-70h] BYREF

  v2 = a1;
  v3 = (__int64 *)sub_B2BE50(*(_QWORD *)a2);
  *(_QWORD *)(a1 + 400) = sub_B0D000(v3, 0, 0, 0, 1);
  v8 = *(__int64 **)(a2 + 328);
  v67 = (__int64 *)(a2 + 320);
  if ( v8 == (__int64 *)(a2 + 320) )
  {
    v14 = a1 + 664;
    v71 = v73;
    v72 = 0x800000000LL;
    sub_37DC210((__int64)&v71, a2, v4, v5, v6);
    v63 = a1 + 696;
LABEL_66:
    ++*(_QWORD *)(v2 + 664);
    ++*(_QWORD *)(v2 + 696);
    goto LABEL_16;
  }
  v9 = 0;
  do
  {
    v10 = (__int64 *)v8[7];
    ++v9;
    if ( v8 + 6 == v10 )
    {
LABEL_7:
      sub_37BC2F0((__int64)&v71, a1 + 440, v8, v5, v6, v7);
    }
    else
    {
      while ( !v10[7] || !(unsigned int)sub_B10CE0((__int64)(v10 + 7)) )
      {
        v10 = (__int64 *)v10[1];
        if ( v8 + 6 == v10 )
          goto LABEL_7;
      }
    }
    v8 = (__int64 *)v8[1];
  }
  while ( v67 != v8 );
  v2 = a1;
  v71 = v73;
  v72 = 0x800000000LL;
  sub_37DC210((__int64)&v71, a2, v11, v5, v6);
  if ( *(_DWORD *)(a1 + 612) < v9 )
    sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v9, 8u, v12, v13);
  v14 = a1 + 664;
  v63 = a1 + 696;
  if ( !v9 )
    goto LABEL_66;
  v15 = 4 * v9 / 3 + 1;
  v16 = sub_AF1560(v15);
  ++*(_QWORD *)(a1 + 664);
  if ( *(_DWORD *)(a1 + 688) < v16 )
    sub_2E515B0(a1 + 664, v16);
  v17 = sub_AF1560(v15);
  ++*(_QWORD *)(a1 + 696);
  if ( *(_DWORD *)(a1 + 720) < v17 )
    sub_A09770(v63, v17);
LABEL_16:
  v18 = (__int64)v71;
  v19 = &v71[8 * (unsigned int)v72];
  v65 = (unsigned __int64)v71;
  if ( v71 != v19 )
  {
    v20 = 0;
    while ( 1 )
    {
      v25 = *(unsigned int *)(v2 + 608);
      v26 = *((_QWORD *)v19 - 1);
      v27 = *(unsigned int *)(v2 + 612);
      v69 = v26;
      if ( v25 + 1 > v27 )
      {
        v62 = v26;
        sub_C8D5F0(v2 + 600, (const void *)(v2 + 616), v25 + 1, 8u, v26, v13);
        v25 = *(unsigned int *)(v2 + 608);
        v26 = v62;
      }
      *(_QWORD *)(*(_QWORD *)(v2 + 600) + 8 * v25) = v26;
      ++*(_DWORD *)(v2 + 608);
      *(_DWORD *)sub_2E51790(v14, &v69) = v20;
      v28 = *(_DWORD *)(v2 + 720);
      v29 = *(_DWORD *)(v69 + 24);
      v68 = v29;
      if ( !v28 )
        break;
      v13 = *(_QWORD *)(v2 + 704);
      v21 = 0;
      v22 = 1;
      v18 = (v28 - 1) & (37 * v29);
      v23 = (int *)(v13 + 8 * v18);
      v12 = (unsigned int)*v23;
      if ( (_DWORD)v12 != v29 )
      {
        while ( (_DWORD)v12 != -1 )
        {
          if ( !v21 && (_DWORD)v12 == -2 )
            v21 = v23;
          v18 = (v28 - 1) & (v22 + (_DWORD)v18);
          v23 = (int *)(v13 + 8LL * (unsigned int)v18);
          v12 = (unsigned int)*v23;
          if ( v29 == (_DWORD)v12 )
            goto LABEL_19;
          ++v22;
        }
        v34 = *(_DWORD *)(v2 + 712);
        if ( !v21 )
          v21 = v23;
        ++*(_QWORD *)(v2 + 696);
        v18 = (unsigned int)(v34 + 1);
        v70 = v21;
        if ( 4 * (int)v18 < 3 * v28 )
        {
          v12 = v28 >> 3;
          if ( v28 - *(_DWORD *)(v2 + 716) - (unsigned int)v18 <= (unsigned int)v12 )
          {
            sub_A09770(v63, v28);
            sub_A1A0F0(v63, &v68, &v70);
            v29 = v68;
            v21 = v70;
            v18 = (unsigned int)(*(_DWORD *)(v2 + 712) + 1);
          }
LABEL_28:
          *(_DWORD *)(v2 + 712) = v18;
          if ( *v21 != -1 )
            --*(_DWORD *)(v2 + 716);
          *v21 = v29;
          v24 = v21 + 1;
          v21[1] = 0;
          goto LABEL_20;
        }
LABEL_25:
        sub_A09770(v63, 2 * v28);
        v30 = *(_DWORD *)(v2 + 720);
        if ( v30 )
        {
          v29 = v68;
          v31 = v30 - 1;
          v13 = *(_QWORD *)(v2 + 704);
          LODWORD(v32) = v31 & (37 * v68);
          v21 = (int *)(v13 + 8LL * (unsigned int)v32);
          v12 = (unsigned int)*v21;
          if ( (_DWORD)v12 == v68 )
          {
LABEL_27:
            v33 = *(_DWORD *)(v2 + 712);
            v70 = v21;
            v18 = (unsigned int)(v33 + 1);
          }
          else
          {
            v60 = 1;
            v61 = 0;
            while ( (_DWORD)v12 != -1 )
            {
              if ( !v61 && (_DWORD)v12 == -2 )
                v61 = v21;
              v32 = v31 & (unsigned int)(v32 + v60);
              v21 = (int *)(v13 + 8 * v32);
              v12 = (unsigned int)*v21;
              if ( v68 == (_DWORD)v12 )
                goto LABEL_27;
              ++v60;
            }
            if ( !v61 )
              v61 = v21;
            v18 = (unsigned int)(*(_DWORD *)(v2 + 712) + 1);
            v70 = v61;
            v21 = v61;
          }
        }
        else
        {
          v57 = *(_DWORD *)(v2 + 712);
          v29 = v68;
          v21 = 0;
          v70 = 0;
          v18 = (unsigned int)(v57 + 1);
        }
        goto LABEL_28;
      }
LABEL_19:
      v24 = v23 + 1;
LABEL_20:
      *v24 = v20;
      v19 -= 8;
      ++v20;
      if ( (_BYTE *)v65 == v19 )
        goto LABEL_43;
    }
    ++*(_QWORD *)(v2 + 696);
    v70 = 0;
    goto LABEL_25;
  }
  v20 = 0;
LABEL_43:
  v35 = *(_QWORD *)(a2 + 328);
  if ( v67 != (__int64 *)v35 )
  {
    v66 = v14;
    v36 = v20;
    while ( 1 )
    {
      v39 = *(_DWORD *)(v2 + 688);
      v40 = *(_QWORD *)(v2 + 672);
      if ( !v39 )
        goto LABEL_48;
      v18 = (unsigned int)(v39 - 1);
      v37 = v18 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v38 = *(_QWORD *)(v40 + 16LL * v37);
      if ( v38 == v35 )
      {
LABEL_46:
        v35 = *(_QWORD *)(v35 + 8);
        if ( v67 == (__int64 *)v35 )
          break;
      }
      else
      {
        v56 = 1;
        while ( v38 != -4096 )
        {
          v12 = (unsigned int)(v56 + 1);
          v37 = v18 & (v56 + v37);
          v38 = *(_QWORD *)(v40 + 16LL * v37);
          if ( v35 == v38 )
            goto LABEL_46;
          ++v56;
        }
LABEL_48:
        v41 = *(unsigned int *)(v2 + 608);
        v42 = *(unsigned int *)(v2 + 612);
        v69 = v35;
        if ( v41 + 1 > v42 )
        {
          sub_C8D5F0(v2 + 600, (const void *)(v2 + 616), v41 + 1, 8u, v12, v13);
          v41 = *(unsigned int *)(v2 + 608);
        }
        *(_QWORD *)(*(_QWORD *)(v2 + 600) + 8 * v41) = v35;
        ++*(_DWORD *)(v2 + 608);
        *(_DWORD *)sub_2E51790(v66, &v69) = v36;
        v43 = *(_DWORD *)(v2 + 720);
        v44 = *(_DWORD *)(v69 + 24);
        v68 = v44;
        if ( !v43 )
        {
          ++*(_QWORD *)(v2 + 696);
          v70 = 0;
          goto LABEL_82;
        }
        v45 = 1;
        v12 = 0;
        v13 = *(_QWORD *)(v2 + 704);
        v18 = (v43 - 1) & (37 * v44);
        v46 = (int *)(v13 + 8 * v18);
        v47 = *v46;
        if ( v44 != *v46 )
        {
          while ( v47 != -1 )
          {
            if ( !v12 && v47 == -2 )
              v12 = (__int64)v46;
            v18 = (v43 - 1) & (v45 + (_DWORD)v18);
            v46 = (int *)(v13 + 8LL * (unsigned int)v18);
            v47 = *v46;
            if ( v44 == *v46 )
              goto LABEL_52;
            ++v45;
          }
          v58 = *(_DWORD *)(v2 + 712);
          if ( v12 )
            v46 = (int *)v12;
          ++*(_QWORD *)(v2 + 696);
          v59 = v58 + 1;
          v70 = v46;
          if ( 4 * (v58 + 1) >= 3 * v43 )
          {
LABEL_82:
            v43 *= 2;
          }
          else
          {
            v18 = v43 - *(_DWORD *)(v2 + 716) - v59;
            v13 = v43 >> 3;
            if ( (unsigned int)v18 > (unsigned int)v13 )
            {
LABEL_78:
              *(_DWORD *)(v2 + 712) = v59;
              if ( *v46 != -1 )
                --*(_DWORD *)(v2 + 716);
              *v46 = v44;
              v46[1] = 0;
              goto LABEL_52;
            }
          }
          sub_A09770(v63, v43);
          sub_A1A0F0(v63, &v68, &v70);
          v44 = v68;
          v59 = *(_DWORD *)(v2 + 712) + 1;
          v46 = v70;
          goto LABEL_78;
        }
LABEL_52:
        v46[1] = v36;
        v35 = *(_QWORD *)(v35 + 8);
        ++v36;
        if ( v67 == (__int64 *)v35 )
          break;
      }
    }
  }
  v48 = *(unsigned int **)(a2 + 904);
  v49 = *(unsigned int *)(a2 + 912);
  v50 = 20 * v49;
  v51 = &v48[5 * v49];
  if ( v48 != v51 )
  {
    v52 = (__m128i *)&v48[5 * v49];
    _BitScanReverse64(&v53, 0xCCCCCCCCCCCCCCCDLL * (v50 >> 2));
    sub_37DD680(*(_QWORD *)(a2 + 904), v52, 2LL * (int)(63 - (v53 ^ 0x3F)), v18, v12, v13);
    if ( (unsigned __int64)v50 <= 0x140 )
    {
      sub_37B66A0(v48, v51);
    }
    else
    {
      v54 = (__int64 *)(v48 + 80);
      sub_37B66A0(v48, v48 + 80);
      if ( v51 != v48 + 80 )
      {
        do
        {
          v55 = v54;
          v54 = (__int64 *)((char *)v54 + 20);
          sub_37B65B0(v55);
        }
        while ( v51 != (unsigned int *)v54 );
      }
    }
  }
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
}
