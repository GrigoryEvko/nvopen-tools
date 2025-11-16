// Function: sub_2B33530
// Address: 0x2b33530
//
_QWORD *__fastcall sub_2B33530(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v3; // ebx
  __int64 v4; // rcx
  int v5; // edx
  unsigned int v6; // ecx
  _QWORD *v7; // rax
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rdx
  unsigned int *i; // rax
  _QWORD *v12; // r15
  __int64 *v13; // r12
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 *v17; // r14
  _BYTE *v18; // rdi
  unsigned __int64 v19; // rax
  _BYTE *v20; // rdi
  unsigned __int64 v21; // rax
  _BYTE *v22; // rdi
  unsigned __int64 v23; // rax
  _BYTE *v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 *v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // r14
  __int64 v32; // r15
  unsigned int v33; // eax
  unsigned int v34; // r10d
  __int64 v35; // r15
  __int64 v37; // rdx
  int v38; // r12d
  __int64 v39; // r14
  __int64 v40; // r12
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rdi
  unsigned __int64 v44; // rax
  __int64 v45; // rdi
  unsigned __int64 v46; // rax
  __int64 v47; // rdi
  unsigned __int64 v48; // rax
  unsigned int v49; // [rsp+Ch] [rbp-164h]
  __int64 *v50; // [rsp+10h] [rbp-160h]
  __int64 v51; // [rsp+18h] [rbp-158h]
  int v52; // [rsp+18h] [rbp-158h]
  __int64 v53; // [rsp+28h] [rbp-148h]
  __int64 v56; // [rsp+40h] [rbp-130h]
  _QWORD *v57; // [rsp+48h] [rbp-128h]
  __int64 v58; // [rsp+50h] [rbp-120h]
  _DWORD *v59; // [rsp+58h] [rbp-118h]
  __int64 v60; // [rsp+68h] [rbp-108h]
  _QWORD v61[4]; // [rsp+70h] [rbp-100h] BYREF
  char v62[32]; // [rsp+90h] [rbp-E0h] BYREF
  __int16 v63; // [rsp+B0h] [rbp-C0h]
  char v64[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v65; // [rsp+E0h] [rbp-90h]
  __m128i v66; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v67; // [rsp+100h] [rbp-70h]
  __int64 v68; // [rsp+108h] [rbp-68h]
  __int64 v69; // [rsp+110h] [rbp-60h]
  __int64 v70; // [rsp+118h] [rbp-58h]
  __int64 v71; // [rsp+120h] [rbp-50h]
  __int64 v72; // [rsp+128h] [rbp-48h]
  __int16 v73; // [rsp+130h] [rbp-40h]

  v53 = **(_QWORD **)a1 + 16LL * *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( **(_QWORD **)a1 != v53 )
  {
    v59 = **(_DWORD ***)a1;
    v57 = a2;
    while ( 1 )
    {
      v3 = v59[2];
      v56 = *(_QWORD *)v59;
      v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v59 + 96LL) + 8LL);
      v58 = *(_QWORD *)(*(_QWORD *)v59 + 96LL);
      v5 = *(unsigned __int8 *)(v4 + 8);
      if ( (unsigned int)(v5 - 17) <= 1 )
        LOBYTE(v5) = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
      if ( (_BYTE)v5 != 12 )
        goto LABEL_7;
      v12 = *(_QWORD **)(a1 + 8);
      v13 = *(__int64 **)v56;
      v14 = 8LL * *(unsigned int *)(v56 + 8);
      v51 = *(_QWORD *)v56 + v14;
      v15 = v14 >> 3;
      v16 = v14 >> 5;
      if ( v16 )
      {
        v17 = &v13[4 * v16];
        while ( 1 )
        {
          v24 = (_BYTE *)*v13;
          if ( *(_BYTE *)*v13 != 13 )
          {
            v25 = *(_QWORD *)(v12[15] + 3344LL);
            v67 = 0;
            v66 = (__m128i)v25;
            v68 = 0;
            v69 = 0;
            v70 = 0;
            v71 = 0;
            v72 = 0;
            v73 = 257;
            if ( !(unsigned __int8)sub_9AC470((__int64)v24, &v66, 0) )
              goto LABEL_26;
          }
          v18 = (_BYTE *)v13[1];
          if ( *v18 != 13 )
          {
            v19 = *(_QWORD *)(v12[15] + 3344LL);
            v67 = 0;
            v66 = (__m128i)v19;
            v68 = 0;
            v69 = 0;
            v70 = 0;
            v71 = 0;
            v72 = 0;
            v73 = 257;
            if ( !(unsigned __int8)sub_9AC470((__int64)v18, &v66, 0) )
            {
              ++v13;
              goto LABEL_26;
            }
          }
          v20 = (_BYTE *)v13[2];
          if ( *v20 != 13 )
          {
            v21 = *(_QWORD *)(v12[15] + 3344LL);
            v67 = 0;
            v66 = (__m128i)v21;
            v68 = 0;
            v69 = 0;
            v70 = 0;
            v71 = 0;
            v72 = 0;
            v73 = 257;
            if ( !(unsigned __int8)sub_9AC470((__int64)v20, &v66, 0) )
            {
              v13 += 2;
              goto LABEL_26;
            }
          }
          v22 = (_BYTE *)v13[3];
          if ( *v22 != 13 )
          {
            v23 = *(_QWORD *)(v12[15] + 3344LL);
            v67 = 0;
            v66 = (__m128i)v23;
            v68 = 0;
            v69 = 0;
            v70 = 0;
            v71 = 0;
            v72 = 0;
            v73 = 257;
            if ( !(unsigned __int8)sub_9AC470((__int64)v22, &v66, 0) )
            {
              v13 += 3;
              goto LABEL_26;
            }
          }
          v13 += 4;
          if ( v17 == v13 )
          {
            v15 = (v51 - (__int64)v13) >> 3;
            break;
          }
        }
      }
      if ( v15 == 2 )
        goto LABEL_57;
      if ( v15 == 3 )
        break;
      if ( v15 == 1 )
        goto LABEL_60;
LABEL_45:
      v13 = (__int64 *)v51;
LABEL_26:
      v26 = *(_QWORD *)(v58 + 8);
      v27 = *v12;
      if ( (unsigned int)*(unsigned __int8 *)(*v12 + 8LL) - 17 <= 1 )
        v27 = **(_QWORD **)(v27 + 16);
      if ( *(_QWORD *)(v26 + 24) != v27 )
      {
        v28 = (__int64 *)v12[14];
        v63 = 257;
        v50 = v28;
        v29 = *(_QWORD *)(v12[15] + 3344LL);
        v73 = 257;
        v66 = (__m128i)v29;
        v67 = 0;
        v68 = 0;
        v69 = 0;
        v70 = 0;
        v71 = 0;
        v72 = 0;
        sub_9AC470(v58, &v66, 0);
        v30 = *v12;
        if ( (unsigned int)*(unsigned __int8 *)(*v12 + 8LL) - 17 <= 1 )
          v30 = **(_QWORD **)(v30 + 16);
        BYTE4(v60) = *(_BYTE *)(v26 + 8) == 18;
        LODWORD(v60) = *(_DWORD *)(v26 + 32);
        v31 = sub_BCE1B0((__int64 *)v30, v60);
        v32 = *(_QWORD *)(v58 + 8);
        v49 = sub_BCB060(v32);
        v33 = sub_BCB060(v31);
        v34 = 38;
        if ( v49 <= v33 )
          v34 = (v51 != (_QWORD)v13) + 39;
        if ( v31 == v32 )
        {
          v35 = v58;
        }
        else
        {
          v52 = v34;
          v35 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v50[10] + 120LL))(
                  v50[10],
                  v34,
                  v58,
                  v31);
          if ( !v35 )
          {
            v65 = 257;
            v35 = sub_B51D30(v52, v58, v31, (__int64)v64, 0, 0);
            if ( (unsigned __int8)sub_920620(v35) )
            {
              v37 = v50[12];
              v38 = *((_DWORD *)v50 + 26);
              if ( v37 )
                sub_B99FD0(v35, 3u, v37);
              sub_B45150(v35, v38);
            }
            (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v50[11] + 16LL))(
              v50[11],
              v35,
              v62,
              v50[7],
              v50[8]);
            v39 = *v50;
            v40 = *v50 + 16LL * *((unsigned int *)v50 + 2);
            if ( *v50 != v40 )
            {
              do
              {
                v41 = *(_QWORD *)(v39 + 8);
                v42 = *(_DWORD *)v39;
                v39 += 16;
                sub_B99FD0(v35, v42, v41);
              }
              while ( v40 != v39 );
            }
          }
        }
        v58 = v35;
      }
LABEL_7:
      v6 = v3;
      v7 = *(_QWORD **)(a1 + 8);
      if ( *(_BYTE *)(*v7 + 8LL) == 17 )
        v6 = *(_DWORD *)(*v7 + 32LL) * v3;
      v8 = v7[14];
      v61[1] = 0;
      v61[2] = v7;
      v61[0] = sub_2B7B290;
      v57 = (_QWORD *)sub_2B330C0(
                        v8,
                        v57,
                        v58,
                        v6,
                        (__int64 (__fastcall *)(__int64, _QWORD *, __int64, void *, _QWORD))sub_2B0B930,
                        (__int64)v61);
      if ( *(_DWORD *)(a3 + 8) )
      {
        v9 = *(_DWORD *)(v56 + 120);
        if ( !v9 )
          v9 = *(_DWORD *)(v56 + 8);
        v10 = *(_QWORD *)a3 + 4LL * (v3 + v9);
        for ( i = (unsigned int *)(*(_QWORD *)a3 + 4LL * v3); i != (unsigned int *)v10; ++v3 )
          *i++ = v3;
      }
      v59 += 4;
      if ( (_DWORD *)v53 == v59 )
        return v57;
    }
    v43 = *v13;
    if ( *(_BYTE *)*v13 != 13 )
    {
      v44 = *(_QWORD *)(v12[15] + 3344LL);
      v67 = 0;
      v66 = (__m128i)v44;
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v72 = 0;
      v73 = 257;
      if ( !(unsigned __int8)sub_9AC470(v43, &v66, 0) )
        goto LABEL_26;
    }
    ++v13;
LABEL_57:
    v45 = *v13;
    if ( *(_BYTE *)*v13 != 13 )
    {
      v46 = *(_QWORD *)(v12[15] + 3344LL);
      v67 = 0;
      v66 = (__m128i)v46;
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v72 = 0;
      v73 = 257;
      if ( !(unsigned __int8)sub_9AC470(v45, &v66, 0) )
        goto LABEL_26;
    }
    ++v13;
LABEL_60:
    v47 = *v13;
    if ( *(_BYTE *)*v13 != 13 )
    {
      v48 = *(_QWORD *)(v12[15] + 3344LL);
      v67 = 0;
      v66 = (__m128i)v48;
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v72 = 0;
      v73 = 257;
      if ( (unsigned __int8)sub_9AC470(v47, &v66, 0) )
        v13 = (__int64 *)v51;
      goto LABEL_26;
    }
    goto LABEL_45;
  }
  return a2;
}
