// Function: sub_359A420
// Address: 0x359a420
//
void __fastcall sub_359A420(__int64 *a1, __int32 a2, __int32 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  _BYTE *v13; // rdi
  __int64 v14; // r12
  int v15; // esi
  __int64 v16; // rcx
  __int32 v17; // r13d
  unsigned int i; // eax
  __int64 v19; // rax
  __int32 v20; // eax
  unsigned __int8 *v21; // rsi
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 *v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int v31; // r15d
  __int64 v32; // rdi
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int32 v35; // eax
  __int64 v36; // rdi
  __int32 v37; // r15d
  __int64 v38; // r13
  __int64 *v39; // rax
  unsigned __int8 *v40; // rax
  unsigned __int8 *v41; // rdx
  __int64 *v42; // r12
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdi
  __int64 *v48; // r12
  __int64 *v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // r12
  unsigned int v52; // eax
  unsigned __int64 v53; // rdx
  unsigned int v54; // eax
  __int64 v55; // rax
  __int64 v56; // r9
  unsigned __int64 v57; // r13
  __int64 *v58; // rcx
  __int64 *v59; // rsi
  _BYTE *v60; // [rsp+10h] [rbp-150h]
  __int64 v61; // [rsp+18h] [rbp-148h]
  __int64 v62; // [rsp+18h] [rbp-148h]
  __int32 v64; // [rsp+2Ch] [rbp-134h]
  _BYTE *v65; // [rsp+38h] [rbp-128h]
  __int64 v66; // [rsp+38h] [rbp-128h]
  __int64 v67; // [rsp+48h] [rbp-118h] BYREF
  unsigned __int8 *v68[2]; // [rsp+50h] [rbp-110h] BYREF
  unsigned __int8 *v69; // [rsp+60h] [rbp-100h] BYREF
  __int64 v70; // [rsp+68h] [rbp-F8h]
  __int64 v71; // [rsp+70h] [rbp-F0h]
  __m128i v72; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v73; // [rsp+90h] [rbp-D0h]
  __int64 v74; // [rsp+98h] [rbp-C8h]
  __int64 v75; // [rsp+A0h] [rbp-C0h]
  __int64 *v76; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v77; // [rsp+B8h] [rbp-A8h]
  _BYTE v78[48]; // [rsp+C0h] [rbp-A0h] BYREF
  _BYTE *v79; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v80; // [rsp+F8h] [rbp-68h]
  _BYTE v81[96]; // [rsp+100h] [rbp-60h] BYREF

  v76 = (__int64 *)v78;
  v7 = a1[3];
  v77 = 0x600000000LL;
  v79 = v81;
  v80 = 0x600000000LL;
  if ( a2 < 0 )
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + 304) + 8LL * (unsigned int)a2);
  if ( !v8 )
    goto LABEL_40;
  if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
  {
LABEL_5:
    v9 = *(_QWORD *)(v8 + 16);
    v10 = *(_QWORD *)(v9 + 24);
    if ( a1[6] == v10 )
      goto LABEL_47;
    if ( a1[10] != v10 && a1[11] != v10 && a1[12] != v10 )
    {
      v11 = (unsigned int)v77;
      v12 = (unsigned int)v77 + 1LL;
      if ( v12 > HIDWORD(v77) )
      {
        sub_C8D5F0((__int64)&v76, v78, v12, 8u, a5, a6);
        v11 = (unsigned int)v77;
      }
      v76[v11] = v8;
      v9 = *(_QWORD *)(v8 + 16);
      LODWORD(v77) = v77 + 1;
      if ( a1[6] == *(_QWORD *)(v9 + 24) )
      {
LABEL_47:
        if ( !*(_WORD *)(v9 + 68) || *(_WORD *)(v9 + 68) == 68 )
        {
          v33 = (unsigned int)v80;
          v34 = (unsigned int)v80 + 1LL;
          if ( v34 > HIDWORD(v80) )
          {
            sub_C8D5F0((__int64)&v79, v81, v34, 8u, a5, a6);
            v33 = (unsigned int)v80;
          }
          *(_QWORD *)&v79[8 * v33] = v9;
          LODWORD(v80) = v80 + 1;
        }
      }
    }
    while ( 1 )
    {
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        break;
      if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
        goto LABEL_5;
    }
    if ( !(_DWORD)v77 )
      goto LABEL_16;
    v35 = sub_2EC06C0(
            a1[3],
            *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
            byte_3F871B3,
            0,
            a5,
            a6);
    v36 = a1[14];
    v67 = 0;
    v37 = v35;
    v38 = *(_QWORD *)(a1[4] + 8);
    v69 = 0;
    v70 = 0;
    v71 = 0;
    v39 = (__int64 *)sub_2E311E0(v36);
    v40 = (unsigned __int8 *)sub_2F26260(a1[14], v39, (__int64 *)&v69, v38, v37);
    v68[1] = v41;
    v68[0] = v40;
    v42 = sub_3598AB0((__int64 *)v68, a2, 0, 0);
    v43 = a1[6];
    v44 = v42[1];
    v72.m128i_i8[0] = 4;
    v74 = v43;
    v72.m128i_i32[0] &= 0xFFF000FF;
    v73 = 0;
    sub_2E8EAD0(v44, *v42, &v72);
    v45 = sub_3598AB0(v42, a3, 0, 0);
    v46 = a1[12];
    v47 = v45[1];
    v72.m128i_i8[0] = 4;
    v74 = v46;
    v72.m128i_i32[0] &= 0xFFF000FF;
    v73 = 0;
    sub_2E8EAD0(v47, *v45, &v72);
    sub_9C6650(&v69);
    sub_9C6650(&v67);
    v48 = v76;
    v49 = &v76[(unsigned int)v77];
    if ( v49 != v76 )
    {
      do
      {
        v50 = *v48++;
        sub_2EAB0C0(v50, v37);
      }
      while ( v49 != v48 );
    }
    v51 = a1[5];
    v52 = v37 & 0x7FFFFFFF;
    v53 = *(unsigned int *)(v51 + 160);
    if ( (v37 & 0x7FFFFFFFu) < (unsigned int)v53 )
    {
      if ( *(_QWORD *)(*(_QWORD *)(v51 + 152) + 8LL * v52) )
        goto LABEL_16;
    }
    v54 = v52 + 1;
    if ( (unsigned int)v53 < v54 && v54 != v53 )
    {
      if ( v54 >= v53 )
      {
        v56 = *(_QWORD *)(v51 + 168);
        v57 = v54 - v53;
        if ( v54 > (unsigned __int64)*(unsigned int *)(v51 + 164) )
        {
          v66 = *(_QWORD *)(v51 + 168);
          sub_C8D5F0(v51 + 152, (const void *)(v51 + 168), v54, 8u, v54, v56);
          v53 = *(unsigned int *)(v51 + 160);
          v56 = v66;
        }
        v55 = *(_QWORD *)(v51 + 152);
        v58 = (__int64 *)(v55 + 8 * v53);
        v59 = &v58[v57];
        if ( v58 != v59 )
        {
          do
            *v58++ = v56;
          while ( v59 != v58 );
          LODWORD(v53) = *(_DWORD *)(v51 + 160);
          v55 = *(_QWORD *)(v51 + 152);
        }
        *(_DWORD *)(v51 + 160) = v57 + v53;
        goto LABEL_60;
      }
      *(_DWORD *)(v51 + 160) = v54;
    }
    v55 = *(_QWORD *)(v51 + 152);
LABEL_60:
    *(_QWORD *)(v55 + 8LL * (v37 & 0x7FFFFFFF)) = sub_2E10F30(v37);
LABEL_16:
    v13 = v79;
    if ( (_DWORD)v80 )
    {
      v65 = v79;
      v60 = &v79[8 * (unsigned int)v80];
      do
      {
        v14 = *(_QWORD *)v65;
        v15 = *(_DWORD *)(*(_QWORD *)v65 + 40LL) & 0xFFFFFF;
        if ( v15 == 1 )
        {
          v19 = 0;
          v17 = 0;
        }
        else
        {
          v16 = *(_QWORD *)(v14 + 32);
          v17 = 0;
          for ( i = 1; i != v15; i += 2 )
          {
            if ( a1[6] != *(_QWORD *)(v16 + 40LL * (i + 1) + 24) )
              v17 = *(_DWORD *)(v16 + 40LL * i + 8);
          }
          v19 = 16LL * (v17 & 0x7FFFFFFF);
        }
        v20 = sub_2EC06C0(
                a1[3],
                *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + v19) & 0xFFFFFFFFFFFFFFF8LL,
                byte_3F871B3,
                0,
                a5,
                a6);
        v21 = *(unsigned __int8 **)(v14 + 56);
        v64 = v20;
        v22 = *(_QWORD *)(a1[4] + 8);
        v68[0] = v21;
        if ( v21 )
        {
          sub_B96E90((__int64)v68, (__int64)v21, 1);
          v69 = v68[0];
          if ( v68[0] )
          {
            sub_B976B0((__int64)v68, v68[0], (__int64)&v69);
            v68[0] = 0;
          }
        }
        else
        {
          v69 = 0;
        }
        v23 = a1[13];
        v70 = 0;
        v71 = 0;
        v24 = (__int64 *)sub_2E311E0(v23);
        v25 = sub_2F26260(a1[13], v24, (__int64 *)&v69, v22, v64);
        v27 = v26;
        v73 = 0;
        v61 = (__int64)v25;
        v72.m128i_i32[2] = v17;
        v74 = 0;
        v75 = 0;
        v72.m128i_i64[0] = 0;
        sub_2E8EAD0(v26, (__int64)v25, &v72);
        v28 = a1[9];
        v72.m128i_i8[0] = 4;
        v74 = v28;
        v72.m128i_i32[0] &= 0xFFF000FF;
        v73 = 0;
        sub_2E8EAD0(v27, v61, &v72);
        v72.m128i_i32[2] = a3;
        v73 = 0;
        v74 = 0;
        v75 = 0;
        v72.m128i_i64[0] = 0;
        sub_2E8EAD0(v27, v61, &v72);
        v29 = a1[12];
        v72.m128i_i8[0] = 4;
        v73 = 0;
        v72.m128i_i32[0] &= 0xFFF000FF;
        v74 = v29;
        sub_2E8EAD0(v27, v61, &v72);
        if ( v69 )
          sub_B91220((__int64)&v69, (__int64)v69);
        if ( v68[0] )
          sub_B91220((__int64)v68, (__int64)v68[0]);
        v30 = *(_DWORD *)(v14 + 40) & 0xFFFFFF;
        if ( v30 > 1 )
        {
          v31 = 1;
          v32 = *(_QWORD *)(v14 + 32) + 40LL;
          while ( v17 != *(_DWORD *)(v32 + 8) )
          {
            v31 += 2;
            v32 += 80;
            if ( v31 >= v30 )
              goto LABEL_36;
          }
          v62 = a1[13];
          sub_2EAB0C0(v32, v64);
          *(_QWORD *)(*(_QWORD *)(v14 + 32) + 40LL * (v31 + 1) + 24) = v62;
        }
LABEL_36:
        v65 += 8;
      }
      while ( v60 != v65 );
      v13 = v79;
    }
    if ( v13 != v81 )
      _libc_free((unsigned __int64)v13);
    goto LABEL_40;
  }
  while ( 1 )
  {
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      break;
    if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
      goto LABEL_5;
  }
LABEL_40:
  if ( v76 != (__int64 *)v78 )
    _libc_free((unsigned __int64)v76);
}
