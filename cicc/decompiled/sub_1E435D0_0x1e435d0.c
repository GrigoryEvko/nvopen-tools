// Function: sub_1E435D0
// Address: 0x1e435d0
//
__int64 __fastcall sub_1E435D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rax
  int v7; // r15d
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rsi
  signed __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // rax
  int v22; // r15d
  int v23; // r14d
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // [rsp+0h] [rbp-100h]
  __int64 v41; // [rsp+8h] [rbp-F8h]
  signed __int64 v42; // [rsp+10h] [rbp-F0h]
  __int64 v43; // [rsp+18h] [rbp-E8h]
  signed __int64 v44; // [rsp+20h] [rbp-E0h]
  int v45; // [rsp+2Ch] [rbp-D4h]
  int v46; // [rsp+2Ch] [rbp-D4h]
  __int64 v47; // [rsp+30h] [rbp-D0h]
  __int64 v48; // [rsp+30h] [rbp-D0h]
  __int64 v49; // [rsp+38h] [rbp-C8h]
  __int64 v50; // [rsp+38h] [rbp-C8h]
  __int64 v51; // [rsp+40h] [rbp-C0h]
  __int64 v52; // [rsp+40h] [rbp-C0h]
  __int64 v53; // [rsp+48h] [rbp-B8h]
  __int64 v54; // [rsp+48h] [rbp-B8h]
  __int64 v55; // [rsp+50h] [rbp-B0h]
  __int64 v56; // [rsp+50h] [rbp-B0h]
  int v57; // [rsp+58h] [rbp-A8h]
  int v58; // [rsp+58h] [rbp-A8h]
  int v59; // [rsp+5Ch] [rbp-A4h]
  int v60; // [rsp+5Ch] [rbp-A4h]
  int v61; // [rsp+60h] [rbp-A0h]
  int v62; // [rsp+60h] [rbp-A0h]
  int v63; // [rsp+64h] [rbp-9Ch]
  int v64; // [rsp+64h] [rbp-9Ch]
  int v65; // [rsp+68h] [rbp-98h]
  int v66; // [rsp+68h] [rbp-98h]
  char v67; // [rsp+6Fh] [rbp-91h]
  char v68; // [rsp+6Fh] [rbp-91h]
  __int64 v69; // [rsp+70h] [rbp-90h] BYREF
  __int64 v70; // [rsp+78h] [rbp-88h]
  int v71; // [rsp+80h] [rbp-80h]
  int v72; // [rsp+84h] [rbp-7Ch]
  int v73; // [rsp+88h] [rbp-78h]
  __int64 v74; // [rsp+90h] [rbp-70h]
  __int64 v75; // [rsp+98h] [rbp-68h]
  __int64 v76; // [rsp+A0h] [rbp-60h]
  char v77; // [rsp+A8h] [rbp-58h]
  int v78; // [rsp+ACh] [rbp-54h]
  int v79; // [rsp+B0h] [rbp-50h]
  int v80; // [rsp+B4h] [rbp-4Ch]
  int v81; // [rsp+B8h] [rbp-48h]
  __int64 v82; // [rsp+C0h] [rbp-40h]
  int v83; // [rsp+C8h] [rbp-38h]

  v41 = a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v40 = a1 + a3 - a2;
  v42 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a1) >> 5);
  if ( 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5) != v42 - 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5) )
  {
    v43 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5);
    while ( 1 )
    {
      v44 = v42 - v43;
      if ( v43 >= v42 - v43 )
      {
        v18 = 96 * v42 + v41;
        v41 = v18 - 96 * v44;
        if ( v43 > 0 )
        {
          v19 = v18 - 96 * v44;
          v20 = 0;
          do
          {
            v21 = *(_QWORD *)(v19 - 88);
            v19 -= 96;
            v22 = *(_DWORD *)(v19 + 20);
            v23 = *(_DWORD *)(v19 + 24);
            ++*(_QWORD *)v19;
            v18 -= 96LL;
            v48 = v21;
            LODWORD(v21) = *(_DWORD *)(v19 + 16);
            *(_DWORD *)(v19 + 20) = 0;
            v46 = v21;
            v24 = *(_QWORD *)(v19 + 32);
            *(_DWORD *)(v19 + 16) = 0;
            v54 = v24;
            v25 = *(_QWORD *)(v19 + 40);
            *(_DWORD *)(v19 + 24) = 0;
            v52 = v25;
            v26 = *(_QWORD *)(v19 + 48);
            *(_QWORD *)(v19 + 40) = 0;
            v50 = v26;
            LOBYTE(v26) = *(_BYTE *)(v19 + 56);
            *(_QWORD *)(v19 + 48) = 0;
            v68 = v26;
            LODWORD(v26) = *(_DWORD *)(v19 + 60);
            *(_QWORD *)(v19 + 32) = 0;
            v66 = v26;
            LODWORD(v26) = *(_DWORD *)(v19 + 64);
            *(_QWORD *)(v19 + 8) = 0;
            v64 = v26;
            v62 = *(_DWORD *)(v19 + 68);
            v60 = *(_DWORD *)(v19 + 72);
            v56 = *(_QWORD *)(v19 + 80);
            v58 = *(_DWORD *)(v19 + 88);
            j___libc_free_0(0);
            ++*(_QWORD *)v19;
            *(_DWORD *)(v19 + 24) = 0;
            *(_QWORD *)(v19 + 8) = 0;
            *(_DWORD *)(v19 + 16) = 0;
            *(_DWORD *)(v19 + 20) = 0;
            v27 = *(_QWORD *)(v18 + 8);
            ++*(_QWORD *)v18;
            v28 = *(_QWORD *)(v19 + 8);
            *(_QWORD *)(v19 + 8) = v27;
            LODWORD(v27) = *(_DWORD *)(v18 + 16);
            *(_QWORD *)(v18 + 8) = v28;
            LODWORD(v28) = *(_DWORD *)(v19 + 16);
            *(_DWORD *)(v19 + 16) = v27;
            LODWORD(v27) = *(_DWORD *)(v18 + 20);
            *(_DWORD *)(v18 + 16) = v28;
            LODWORD(v28) = *(_DWORD *)(v19 + 20);
            *(_DWORD *)(v19 + 20) = v27;
            LODWORD(v27) = *(_DWORD *)(v18 + 24);
            *(_DWORD *)(v18 + 20) = v28;
            LODWORD(v28) = *(_DWORD *)(v19 + 24);
            *(_DWORD *)(v19 + 24) = v27;
            *(_DWORD *)(v18 + 24) = v28;
            v29 = *(_QWORD *)(v19 + 32);
            v30 = *(_QWORD *)(v19 + 48);
            *(_QWORD *)(v19 + 32) = *(_QWORD *)(v18 + 32);
            *(_QWORD *)(v19 + 40) = *(_QWORD *)(v18 + 40);
            *(_QWORD *)(v19 + 48) = *(_QWORD *)(v18 + 48);
            *(_QWORD *)(v18 + 32) = 0;
            *(_QWORD *)(v18 + 40) = 0;
            *(_QWORD *)(v18 + 48) = 0;
            if ( v29 )
              j_j___libc_free_0(v29, v30 - v29);
            *(_BYTE *)(v19 + 56) = *(_BYTE *)(v18 + 56);
            *(_DWORD *)(v19 + 60) = *(_DWORD *)(v18 + 60);
            *(_DWORD *)(v19 + 64) = *(_DWORD *)(v18 + 64);
            *(_DWORD *)(v19 + 68) = *(_DWORD *)(v18 + 68);
            *(_DWORD *)(v19 + 72) = *(_DWORD *)(v18 + 72);
            *(_QWORD *)(v19 + 80) = *(_QWORD *)(v18 + 80);
            *(_DWORD *)(v19 + 88) = *(_DWORD *)(v18 + 88);
            j___libc_free_0(*(_QWORD *)(v18 + 8));
            v31 = *(_QWORD *)(v18 + 32);
            *(_DWORD *)(v18 + 20) = v22;
            v32 = *(_QWORD *)(v18 + 48);
            ++*(_QWORD *)v18;
            *(_QWORD *)(v18 + 8) = v48;
            *(_DWORD *)(v18 + 24) = v23;
            *(_DWORD *)(v18 + 16) = v46;
            *(_QWORD *)(v18 + 32) = v54;
            *(_QWORD *)(v18 + 40) = v52;
            *(_QWORD *)(v18 + 48) = v50;
            if ( v31 )
              j_j___libc_free_0(v31, v32 - v31);
            ++v20;
            *(_BYTE *)(v18 + 56) = v68;
            *(_DWORD *)(v18 + 60) = v66;
            *(_DWORD *)(v18 + 64) = v64;
            *(_DWORD *)(v18 + 68) = v62;
            *(_DWORD *)(v18 + 72) = v60;
            *(_QWORD *)(v18 + 80) = v56;
            *(_DWORD *)(v18 + 88) = v58;
            j___libc_free_0(0);
          }
          while ( v43 != v20 );
          v41 += -96 * v43;
        }
        v43 = v42 % v44;
        if ( !(v42 % v44) )
          return v40;
      }
      else
      {
        v3 = v41;
        v4 = v41 + 96 * v43;
        if ( v42 - v43 > 0 )
        {
          v5 = 0;
          do
          {
            v6 = *(_QWORD *)(v3 + 8);
            v7 = *(_DWORD *)(v3 + 20);
            *(_QWORD *)(v3 + 8) = 0;
            v8 = *(_DWORD *)(v3 + 24);
            ++*(_QWORD *)v3;
            v47 = v6;
            LODWORD(v6) = *(_DWORD *)(v3 + 16);
            *(_DWORD *)(v3 + 20) = 0;
            v45 = v6;
            v9 = *(_QWORD *)(v3 + 32);
            *(_DWORD *)(v3 + 16) = 0;
            v53 = v9;
            v10 = *(_QWORD *)(v3 + 40);
            *(_DWORD *)(v3 + 24) = 0;
            v51 = v10;
            v11 = *(_QWORD *)(v3 + 48);
            *(_QWORD *)(v3 + 40) = 0;
            v49 = v11;
            LOBYTE(v11) = *(_BYTE *)(v3 + 56);
            *(_QWORD *)(v3 + 48) = 0;
            v67 = v11;
            LODWORD(v11) = *(_DWORD *)(v3 + 60);
            *(_QWORD *)(v3 + 32) = 0;
            v65 = v11;
            v63 = *(_DWORD *)(v3 + 64);
            v61 = *(_DWORD *)(v3 + 68);
            v59 = *(_DWORD *)(v3 + 72);
            v55 = *(_QWORD *)(v3 + 80);
            v57 = *(_DWORD *)(v3 + 88);
            j___libc_free_0(0);
            ++*(_QWORD *)v3;
            *(_DWORD *)(v3 + 24) = 0;
            *(_QWORD *)(v3 + 8) = 0;
            *(_DWORD *)(v3 + 16) = 0;
            *(_DWORD *)(v3 + 20) = 0;
            v12 = *(_QWORD *)(v4 + 8);
            ++*(_QWORD *)v4;
            v13 = *(_QWORD *)(v3 + 8);
            *(_QWORD *)(v3 + 8) = v12;
            LODWORD(v12) = *(_DWORD *)(v4 + 16);
            *(_QWORD *)(v4 + 8) = v13;
            LODWORD(v13) = *(_DWORD *)(v3 + 16);
            *(_DWORD *)(v3 + 16) = v12;
            LODWORD(v12) = *(_DWORD *)(v4 + 20);
            *(_DWORD *)(v4 + 16) = v13;
            LODWORD(v13) = *(_DWORD *)(v3 + 20);
            *(_DWORD *)(v3 + 20) = v12;
            LODWORD(v12) = *(_DWORD *)(v4 + 24);
            *(_DWORD *)(v4 + 20) = v13;
            LODWORD(v13) = *(_DWORD *)(v3 + 24);
            *(_DWORD *)(v3 + 24) = v12;
            *(_DWORD *)(v4 + 24) = v13;
            v14 = *(_QWORD *)(v3 + 32);
            v15 = *(_QWORD *)(v3 + 48);
            *(_QWORD *)(v3 + 32) = *(_QWORD *)(v4 + 32);
            *(_QWORD *)(v3 + 40) = *(_QWORD *)(v4 + 40);
            *(_QWORD *)(v3 + 48) = *(_QWORD *)(v4 + 48);
            *(_QWORD *)(v4 + 32) = 0;
            *(_QWORD *)(v4 + 40) = 0;
            *(_QWORD *)(v4 + 48) = 0;
            if ( v14 )
              j_j___libc_free_0(v14, v15 - v14);
            *(_BYTE *)(v3 + 56) = *(_BYTE *)(v4 + 56);
            *(_DWORD *)(v3 + 60) = *(_DWORD *)(v4 + 60);
            *(_DWORD *)(v3 + 64) = *(_DWORD *)(v4 + 64);
            *(_DWORD *)(v3 + 68) = *(_DWORD *)(v4 + 68);
            *(_DWORD *)(v3 + 72) = *(_DWORD *)(v4 + 72);
            *(_QWORD *)(v3 + 80) = *(_QWORD *)(v4 + 80);
            *(_DWORD *)(v3 + 88) = *(_DWORD *)(v4 + 88);
            j___libc_free_0(*(_QWORD *)(v4 + 8));
            v16 = *(_QWORD *)(v4 + 32);
            *(_DWORD *)(v4 + 20) = v7;
            v17 = *(_QWORD *)(v4 + 48);
            ++*(_QWORD *)v4;
            *(_QWORD *)(v4 + 8) = v47;
            *(_DWORD *)(v4 + 24) = v8;
            *(_DWORD *)(v4 + 16) = v45;
            *(_QWORD *)(v4 + 32) = v53;
            *(_QWORD *)(v4 + 40) = v51;
            *(_QWORD *)(v4 + 48) = v49;
            if ( v16 )
              j_j___libc_free_0(v16, v17 - v16);
            v3 += 96;
            v4 += 96;
            ++v5;
            *(_BYTE *)(v4 - 40) = v67;
            *(_DWORD *)(v4 - 36) = v65;
            *(_DWORD *)(v4 - 32) = v63;
            *(_DWORD *)(v4 - 28) = v61;
            *(_DWORD *)(v4 - 24) = v59;
            *(_QWORD *)(v4 - 16) = v55;
            *(_DWORD *)(v4 - 8) = v57;
            j___libc_free_0(0);
          }
          while ( v44 != v5 );
          v41 += 96 * v44;
        }
        if ( !(v42 % v43) )
          return v40;
        v44 = v43;
        v43 -= v42 % v43;
      }
      v42 = v44;
    }
  }
  v34 = a2;
  v35 = a1;
  do
  {
    v36 = *(_QWORD *)(v35 + 8);
    ++*(_QWORD *)v35;
    v69 = 1;
    v70 = v36;
    LODWORD(v36) = *(_DWORD *)(v35 + 16);
    *(_QWORD *)(v35 + 8) = 0;
    v71 = v36;
    LODWORD(v36) = *(_DWORD *)(v35 + 20);
    *(_DWORD *)(v35 + 16) = 0;
    v72 = v36;
    LODWORD(v36) = *(_DWORD *)(v35 + 24);
    *(_DWORD *)(v35 + 20) = 0;
    v73 = v36;
    v37 = *(_QWORD *)(v35 + 32);
    *(_DWORD *)(v35 + 24) = 0;
    v74 = v37;
    v38 = *(_QWORD *)(v35 + 40);
    *(_QWORD *)(v35 + 32) = 0;
    v75 = v38;
    v39 = *(_QWORD *)(v35 + 48);
    *(_QWORD *)(v35 + 40) = 0;
    v76 = v39;
    LOBYTE(v39) = *(_BYTE *)(v35 + 56);
    *(_QWORD *)(v35 + 48) = 0;
    v77 = v39;
    v78 = *(_DWORD *)(v35 + 60);
    v79 = *(_DWORD *)(v35 + 64);
    v80 = *(_DWORD *)(v35 + 68);
    v81 = *(_DWORD *)(v35 + 72);
    v82 = *(_QWORD *)(v35 + 80);
    v83 = *(_DWORD *)(v35 + 88);
    sub_1E41D50(v35, v34);
    sub_1E41D50(v34, (__int64)&v69);
    if ( v74 )
      j_j___libc_free_0(v74, v76 - v74);
    v35 += 96;
    v34 += 96;
    j___libc_free_0(v70);
  }
  while ( a2 != v35 );
  return v35;
}
