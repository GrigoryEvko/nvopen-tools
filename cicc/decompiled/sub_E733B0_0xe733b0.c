// Function: sub_E733B0
// Address: 0xe733b0
//
void __fastcall sub_E733B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdi
  __int64 v4; // r15
  char v5; // al
  unsigned __int64 v6; // r14
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  __int64 i; // r12
  __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rdi
  _QWORD *v17; // r13
  _QWORD *v18; // rbx
  __int64 v19; // r14
  _QWORD *v20; // r15
  _QWORD *v21; // rdi
  __int64 v22; // rdi
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  __int64 j; // r12
  _QWORD *v26; // r15
  _QWORD *v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rdi
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // r14
  _QWORD *k; // r15
  _QWORD *v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // [rsp+8h] [rbp-108h]
  __int64 v39; // [rsp+8h] [rbp-108h]
  int v41; // [rsp+24h] [rbp-ECh]
  int v42; // [rsp+28h] [rbp-E8h]
  int v43; // [rsp+2Ch] [rbp-E4h]
  __int64 v44; // [rsp+30h] [rbp-E0h]
  __int64 v45; // [rsp+38h] [rbp-D8h]
  __int64 v46; // [rsp+40h] [rbp-D0h]
  __int64 v47; // [rsp+48h] [rbp-C8h]
  __int64 v48; // [rsp+50h] [rbp-C0h]
  __int64 v49; // [rsp+58h] [rbp-B8h]
  int v50; // [rsp+60h] [rbp-B0h]
  char v51; // [rsp+64h] [rbp-ACh]
  char v52; // [rsp+65h] [rbp-ABh]
  char v53; // [rsp+66h] [rbp-AAh]
  char v54; // [rsp+67h] [rbp-A9h]
  __int64 v55; // [rsp+68h] [rbp-A8h]
  __int64 v56; // [rsp+70h] [rbp-A0h]
  __int64 v57; // [rsp+78h] [rbp-98h]
  __int64 v58; // [rsp+78h] [rbp-98h]
  _QWORD v59[4]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v60; // [rsp+A0h] [rbp-70h]
  __int64 v61; // [rsp+A8h] [rbp-68h]
  __int64 v62; // [rsp+B0h] [rbp-60h]
  int v63; // [rsp+B8h] [rbp-58h]
  int v64; // [rsp+BCh] [rbp-54h]
  int v65; // [rsp+C0h] [rbp-50h]
  __int64 v66; // [rsp+C8h] [rbp-48h]
  char v67; // [rsp+D0h] [rbp-40h]
  char v68; // [rsp+D1h] [rbp-3Fh]
  int v69; // [rsp+D4h] [rbp-3Ch]
  char v70; // [rsp+D8h] [rbp-38h]
  char v71; // [rsp+D9h] [rbp-37h]

  if ( a1 != a2 && a1 + 96 != a2 )
  {
    v2 = a1 + 96;
    do
    {
      v3 = v2;
      v4 = v2;
      v2 += 96;
      v5 = sub_E72550(v3, a1);
      v55 = *(_QWORD *)(v2 - 96);
      v56 = *(_QWORD *)(v2 - 88);
      v44 = *(_QWORD *)(v2 - 80);
      v46 = *(_QWORD *)(v2 - 72);
      v47 = *(_QWORD *)(v2 - 64);
      v45 = *(_QWORD *)(v2 - 56);
      v49 = *(_QWORD *)(v2 - 48);
      v43 = *(_DWORD *)(v2 - 40);
      v50 = *(_DWORD *)(v2 - 36);
      v41 = *(_DWORD *)(v2 - 32);
      v48 = *(_QWORD *)(v2 - 24);
      v51 = *(_BYTE *)(v2 - 16);
      v52 = *(_BYTE *)(v2 - 15);
      v42 = *(_DWORD *)(v2 - 12);
      v53 = *(_BYTE *)(v2 - 8);
      v54 = *(_BYTE *)(v2 - 7);
      if ( v5 )
      {
        *(_QWORD *)(v2 - 48) = 0;
        *(_QWORD *)(v2 - 56) = 0;
        *(_QWORD *)(v2 - 64) = 0;
        v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - a1) >> 5);
        if ( v4 - a1 > 0 )
        {
          v57 = 0;
          v7 = 0;
          v8 = 0;
          v38 = v2;
          for ( i = v4; ; v57 = *(_QWORD *)(i + 48) )
          {
            v10 = *(_QWORD *)(i - 96);
            i -= 96;
            v11 = v8;
            *(_QWORD *)(i + 96) = v10;
            *(_QWORD *)(i + 104) = *(_QWORD *)(i + 8);
            *(_QWORD *)(i + 112) = *(_QWORD *)(i + 16);
            *(_QWORD *)(i + 120) = *(_QWORD *)(i + 24);
            v12 = *(_QWORD *)(i + 32);
            *(_QWORD *)(i + 32) = 0;
            *(_QWORD *)(i + 128) = v12;
            v13 = *(_QWORD *)(i + 40);
            *(_QWORD *)(i + 40) = 0;
            *(_QWORD *)(i + 136) = v13;
            v14 = *(_QWORD *)(i + 48);
            *(_QWORD *)(i + 48) = 0;
            for ( *(_QWORD *)(i + 144) = v14; v7 != v11; v11 += 13 )
            {
              v15 = (_QWORD *)v11[9];
              if ( v15 != v11 + 11 )
                j_j___libc_free_0(v15, v11[11] + 1LL);
              v16 = v11[6];
              if ( v16 )
                j_j___libc_free_0(v16, v11[8] - v16);
            }
            if ( v8 )
              j_j___libc_free_0(v8, v57 - (_QWORD)v8);
            *(_DWORD *)(i + 152) = *(_DWORD *)(i + 56);
            *(_DWORD *)(i + 156) = *(_DWORD *)(i + 60);
            *(_DWORD *)(i + 160) = *(_DWORD *)(i + 64);
            *(_QWORD *)(i + 168) = *(_QWORD *)(i + 72);
            *(_BYTE *)(i + 176) = *(_BYTE *)(i + 80);
            *(_BYTE *)(i + 177) = *(_BYTE *)(i + 81);
            *(_DWORD *)(i + 180) = *(_DWORD *)(i + 84);
            *(_BYTE *)(i + 184) = *(_BYTE *)(i + 88);
            *(_BYTE *)(i + 185) = *(_BYTE *)(i + 89);
            if ( !--v6 )
              break;
            v8 = *(_QWORD **)(i + 32);
            v7 = *(_QWORD **)(i + 40);
          }
          v2 = v38;
        }
        v17 = *(_QWORD **)(a1 + 32);
        v18 = *(_QWORD **)(a1 + 40);
        *(_QWORD *)a1 = v55;
        v19 = *(_QWORD *)(a1 + 48);
        *(_QWORD *)(a1 + 8) = v56;
        v20 = v17;
        *(_QWORD *)(a1 + 16) = v44;
        *(_QWORD *)(a1 + 24) = v46;
        *(_QWORD *)(a1 + 32) = v47;
        *(_QWORD *)(a1 + 40) = v45;
        for ( *(_QWORD *)(a1 + 48) = v49; v18 != v20; v20 += 13 )
        {
          v21 = (_QWORD *)v20[9];
          if ( v21 != v20 + 11 )
            j_j___libc_free_0(v21, v20[11] + 1LL);
          v22 = v20[6];
          if ( v22 )
            j_j___libc_free_0(v22, v20[8] - v22);
        }
        if ( v17 )
          j_j___libc_free_0(v17, v19 - (_QWORD)v17);
        *(_DWORD *)(a1 + 56) = v43;
        *(_DWORD *)(a1 + 60) = v50;
        *(_DWORD *)(a1 + 64) = v41;
        *(_QWORD *)(a1 + 72) = v48;
        *(_BYTE *)(a1 + 80) = v51;
        *(_BYTE *)(a1 + 81) = v52;
        *(_DWORD *)(a1 + 84) = v42;
        *(_BYTE *)(a1 + 88) = v53;
        *(_BYTE *)(a1 + 89) = v54;
      }
      else
      {
        v23 = 0;
        *(_QWORD *)(v2 - 48) = 0;
        *(_QWORD *)(v2 - 56) = 0;
        v24 = 0;
        v59[0] = v55;
        *(_QWORD *)(v2 - 64) = 0;
        v59[1] = v56;
        v59[2] = v44;
        v59[3] = v46;
        v60 = v47;
        v61 = v45;
        v62 = v49;
        v63 = v43;
        v64 = v50;
        v65 = v41;
        v66 = v48;
        v67 = v51;
        v68 = v52;
        v69 = v42;
        v70 = v53;
        v58 = 0;
        v71 = v54;
        v39 = v2;
        for ( j = v2 - 192; ; v58 = *(_QWORD *)(j + 144) )
        {
          v26 = (_QWORD *)(j + 96);
          if ( !(unsigned __int8)sub_E72550((__int64)v59, j) )
            break;
          v27 = v24;
          *(_QWORD *)(j + 96) = *(_QWORD *)j;
          *(_QWORD *)(j + 104) = *(_QWORD *)(j + 8);
          *(_QWORD *)(j + 112) = *(_QWORD *)(j + 16);
          *(_QWORD *)(j + 120) = *(_QWORD *)(j + 24);
          v28 = *(_QWORD *)(j + 32);
          *(_QWORD *)(j + 32) = 0;
          *(_QWORD *)(j + 128) = v28;
          v29 = *(_QWORD *)(j + 40);
          *(_QWORD *)(j + 40) = 0;
          *(_QWORD *)(j + 136) = v29;
          v30 = *(_QWORD *)(j + 48);
          *(_QWORD *)(j + 48) = 0;
          for ( *(_QWORD *)(j + 144) = v30; v27 != v23; v27 += 13 )
          {
            v31 = (_QWORD *)v27[9];
            if ( v31 != v27 + 11 )
              j_j___libc_free_0(v31, v27[11] + 1LL);
            v32 = v27[6];
            if ( v32 )
              j_j___libc_free_0(v32, v27[8] - v32);
          }
          if ( v24 )
            j_j___libc_free_0(v24, v58 - (_QWORD)v24);
          v33 = *(_DWORD *)(j + 56);
          v24 = *(_QWORD **)(j + 32);
          j -= 96;
          v23 = *(_QWORD **)(j + 136);
          *(_DWORD *)(j + 248) = v33;
          *(_DWORD *)(j + 252) = *(_DWORD *)(j + 156);
          *(_DWORD *)(j + 256) = *(_DWORD *)(j + 160);
          *(_QWORD *)(j + 264) = *(_QWORD *)(j + 168);
          *(_BYTE *)(j + 272) = *(_BYTE *)(j + 176);
          *(_BYTE *)(j + 273) = *(_BYTE *)(j + 177);
          *(_DWORD *)(j + 276) = *(_DWORD *)(j + 180);
          *(_BYTE *)(j + 280) = *(_BYTE *)(j + 184);
          *(_BYTE *)(j + 281) = *(_BYTE *)(j + 185);
        }
        v34 = j + 96;
        v60 = 0;
        v61 = 0;
        v2 = v39;
        *v26 = v55;
        v62 = 0;
        v26[1] = v56;
        v26[2] = v44;
        v26[3] = v46;
        v26[4] = v47;
        v26[5] = v45;
        v26[6] = v49;
        for ( k = v24; k != v23; k += 13 )
        {
          v36 = (_QWORD *)k[9];
          if ( v36 != k + 11 )
            j_j___libc_free_0(v36, k[11] + 1LL);
          v37 = k[6];
          if ( v37 )
            j_j___libc_free_0(v37, k[8] - v37);
        }
        if ( v24 )
          j_j___libc_free_0(v24, v58 - (_QWORD)v24);
        *(_DWORD *)(v34 + 56) = v43;
        *(_DWORD *)(v34 + 60) = v50;
        *(_DWORD *)(v34 + 64) = v41;
        *(_QWORD *)(v34 + 72) = v48;
        *(_BYTE *)(v34 + 80) = v51;
        *(_BYTE *)(v34 + 81) = v52;
        *(_DWORD *)(v34 + 84) = v42;
        *(_BYTE *)(v34 + 88) = v53;
        *(_BYTE *)(v34 + 89) = v54;
      }
    }
    while ( a2 != v2 );
  }
}
