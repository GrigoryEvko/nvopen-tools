// Function: sub_2646940
// Address: 0x2646940
//
__int64 __fastcall sub_2646940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // r8
  int v11; // r15d
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // r11
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r10
  __int64 v28; // r9
  __int64 v29; // r8
  __int64 v30; // r11
  __int64 v31; // rax
  int v32; // r15d
  int v33; // r14d
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // [rsp+0h] [rbp-E0h]
  __int64 v49; // [rsp+8h] [rbp-D8h]
  signed __int64 v50; // [rsp+10h] [rbp-D0h]
  signed __int64 v51; // [rsp+18h] [rbp-C8h]
  __int64 v52; // [rsp+20h] [rbp-C0h]
  __int64 v53; // [rsp+28h] [rbp-B8h]
  __int64 v54; // [rsp+28h] [rbp-B8h]
  __int64 v55; // [rsp+30h] [rbp-B0h]
  __int64 v56; // [rsp+30h] [rbp-B0h]
  __int64 v57; // [rsp+38h] [rbp-A8h]
  __int64 v58; // [rsp+38h] [rbp-A8h]
  __int64 v59; // [rsp+40h] [rbp-A0h]
  __int64 v60; // [rsp+40h] [rbp-A0h]
  int v61; // [rsp+4Ch] [rbp-94h]
  int v62; // [rsp+4Ch] [rbp-94h]
  __int64 v63; // [rsp+50h] [rbp-90h]
  __int64 v64; // [rsp+50h] [rbp-90h]
  __int64 v65; // [rsp+58h] [rbp-88h]
  __int64 v66; // [rsp+58h] [rbp-88h]
  __int64 v67; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v68; // [rsp+68h] [rbp-78h]
  __int64 v69; // [rsp+70h] [rbp-70h]
  __int64 v70; // [rsp+78h] [rbp-68h]
  __int64 v71; // [rsp+80h] [rbp-60h]
  _QWORD v72[2]; // [rsp+88h] [rbp-58h] BYREF
  int v73; // [rsp+98h] [rbp-48h]
  int v74; // [rsp+9Ch] [rbp-44h]
  int v75; // [rsp+A0h] [rbp-40h]

  result = a3;
  v49 = a1;
  if ( a1 != a2 )
  {
    result = a1;
    if ( a2 != a3 )
    {
      v50 = 0x8E38E38E38E38E39LL * ((a3 - a1) >> 3);
      v48 = a1 + a3 - a2;
      if ( 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3) != v50 - 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3) )
      {
        v52 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
        while ( 1 )
        {
          v51 = v50 - v52;
          if ( v52 >= v50 - v52 )
          {
            v22 = v49 + 72 * v50;
            v49 = v22 - 72 * v51;
            if ( v52 > 0 )
            {
              v23 = 0;
              v24 = v22 - 72 * v51 - 72;
              v25 = v22 - 72;
              do
              {
                v26 = *(_QWORD *)(v24 + 32);
                v27 = *(_QWORD *)(v24 + 8);
                *(_QWORD *)(v24 + 8) = 0;
                v28 = *(_QWORD *)(v24 + 16);
                ++*(_QWORD *)(v24 + 40);
                v29 = *(_QWORD *)(v24 + 24);
                v30 = *(_QWORD *)v24;
                v66 = v26;
                v31 = *(_QWORD *)(v24 + 48);
                v32 = *(_DWORD *)(v24 + 60);
                *(_QWORD *)(v24 + 16) = 0;
                *(_QWORD *)(v24 + 24) = 0;
                v33 = *(_DWORD *)(v24 + 64);
                *(_QWORD *)(v24 + 48) = 0;
                *(_DWORD *)(v24 + 60) = 0;
                *(_DWORD *)(v24 + 64) = 0;
                v64 = v31;
                LODWORD(v31) = *(_DWORD *)(v24 + 56);
                *(_DWORD *)(v24 + 56) = 0;
                v54 = v30;
                *(_QWORD *)v24 = *(_QWORD *)v25;
                v56 = v27;
                *(_QWORD *)(v24 + 8) = *(_QWORD *)(v25 + 8);
                v58 = v28;
                *(_QWORD *)(v24 + 16) = *(_QWORD *)(v25 + 16);
                v60 = v29;
                v62 = v31;
                *(_QWORD *)(v24 + 24) = *(_QWORD *)(v25 + 24);
                v34 = *(_QWORD *)(v25 + 32);
                *(_QWORD *)(v25 + 8) = 0;
                *(_QWORD *)(v25 + 16) = 0;
                *(_QWORD *)(v25 + 24) = 0;
                v35 = *(unsigned int *)(v24 + 64);
                *(_QWORD *)(v24 + 32) = v34;
                sub_C7D6A0(*(_QWORD *)(v24 + 48), 4 * v35, 4);
                ++*(_QWORD *)(v24 + 40);
                *(_DWORD *)(v24 + 64) = 0;
                *(_QWORD *)(v24 + 48) = 0;
                *(_DWORD *)(v24 + 56) = 0;
                *(_DWORD *)(v24 + 60) = 0;
                v36 = *(_QWORD *)(v25 + 48);
                ++*(_QWORD *)(v25 + 40);
                v37 = *(_QWORD *)(v24 + 48);
                *(_QWORD *)(v24 + 48) = v36;
                LODWORD(v36) = *(_DWORD *)(v25 + 56);
                *(_QWORD *)(v25 + 48) = v37;
                LODWORD(v37) = *(_DWORD *)(v24 + 56);
                *(_DWORD *)(v24 + 56) = v36;
                LODWORD(v36) = *(_DWORD *)(v25 + 60);
                *(_DWORD *)(v25 + 56) = v37;
                LODWORD(v37) = *(_DWORD *)(v24 + 60);
                *(_DWORD *)(v24 + 60) = v36;
                LODWORD(v36) = *(_DWORD *)(v25 + 64);
                *(_DWORD *)(v25 + 60) = v37;
                LODWORD(v37) = *(_DWORD *)(v24 + 64);
                *(_DWORD *)(v24 + 64) = v36;
                v38 = *(_QWORD *)(v25 + 8);
                *(_DWORD *)(v25 + 64) = v37;
                *(_QWORD *)v25 = v54;
                *(_QWORD *)(v25 + 8) = v56;
                *(_QWORD *)(v25 + 16) = v58;
                *(_QWORD *)(v25 + 24) = v60;
                if ( v38 )
                  j_j___libc_free_0(v38);
                v39 = *(_QWORD *)(v25 + 48);
                ++v23;
                v24 -= 72;
                v40 = 4LL * *(unsigned int *)(v25 + 64);
                *(_QWORD *)(v25 + 32) = v66;
                sub_C7D6A0(v39, v40, 4);
                ++*(_QWORD *)(v25 + 40);
                v25 -= 72;
                *(_QWORD *)(v25 + 120) = v64;
                *(_DWORD *)(v25 + 132) = v32;
                *(_DWORD *)(v25 + 128) = v62;
                *(_DWORD *)(v25 + 136) = v33;
                sub_C7D6A0(0, 0, 4);
              }
              while ( v52 != v23 );
              v49 += -72 * v52;
            }
            v52 = v50 % v51;
            if ( !(v50 % v51) )
              return v48;
          }
          else
          {
            if ( v50 - v52 > 0 )
            {
              v4 = 0;
              v5 = v49 + 72 * v52;
              v6 = (__int64 *)v49;
              do
              {
                v7 = v6[4];
                v8 = v6[1];
                v6[1] = 0;
                v9 = v6[2];
                v10 = v6[3];
                v6[2] = 0;
                v11 = *((_DWORD *)v6 + 15);
                v12 = *((_DWORD *)v6 + 16);
                v65 = v7;
                v13 = v6[6];
                ++v6[5];
                v6[3] = 0;
                v14 = *v6;
                v6[6] = 0;
                *((_DWORD *)v6 + 15) = 0;
                *((_DWORD *)v6 + 16) = 0;
                v63 = v13;
                LODWORD(v13) = *((_DWORD *)v6 + 14);
                *((_DWORD *)v6 + 14) = 0;
                v53 = v14;
                *v6 = *(_QWORD *)v5;
                v55 = v8;
                v6[1] = *(_QWORD *)(v5 + 8);
                v57 = v9;
                v6[2] = *(_QWORD *)(v5 + 16);
                v59 = v10;
                v61 = v13;
                v6[3] = *(_QWORD *)(v5 + 24);
                v15 = *(_QWORD *)(v5 + 32);
                *(_QWORD *)(v5 + 8) = 0;
                *(_QWORD *)(v5 + 16) = 0;
                *(_QWORD *)(v5 + 24) = 0;
                v16 = *((unsigned int *)v6 + 16);
                v6[4] = v15;
                sub_C7D6A0(v6[6], 4 * v16, 4);
                *((_DWORD *)v6 + 16) = 0;
                v6[6] = 0;
                *((_DWORD *)v6 + 14) = 0;
                *((_DWORD *)v6 + 15) = 0;
                ++v6[5];
                v17 = *(_QWORD *)(v5 + 48);
                ++*(_QWORD *)(v5 + 40);
                v18 = v6[6];
                v6[6] = v17;
                LODWORD(v17) = *(_DWORD *)(v5 + 56);
                *(_QWORD *)(v5 + 48) = v18;
                LODWORD(v18) = *((_DWORD *)v6 + 14);
                *((_DWORD *)v6 + 14) = v17;
                LODWORD(v17) = *(_DWORD *)(v5 + 60);
                *(_DWORD *)(v5 + 56) = v18;
                LODWORD(v18) = *((_DWORD *)v6 + 15);
                *((_DWORD *)v6 + 15) = v17;
                *(_DWORD *)(v5 + 60) = v18;
                v19 = *((unsigned int *)v6 + 16);
                *((_DWORD *)v6 + 16) = *(_DWORD *)(v5 + 64);
                v20 = *(_QWORD *)(v5 + 8);
                *(_DWORD *)(v5 + 64) = v19;
                *(_QWORD *)v5 = v53;
                *(_QWORD *)(v5 + 8) = v55;
                *(_QWORD *)(v5 + 16) = v57;
                *(_QWORD *)(v5 + 24) = v59;
                if ( v20 )
                {
                  j_j___libc_free_0(v20);
                  v19 = *(unsigned int *)(v5 + 64);
                }
                v21 = *(_QWORD *)(v5 + 48);
                ++v4;
                v6 += 9;
                *(_QWORD *)(v5 + 32) = v65;
                sub_C7D6A0(v21, 4 * v19, 4);
                ++*(_QWORD *)(v5 + 40);
                v5 += 72;
                *(_QWORD *)(v5 - 24) = v63;
                *(_DWORD *)(v5 - 12) = v11;
                *(_DWORD *)(v5 - 16) = v61;
                *(_DWORD *)(v5 - 8) = v12;
                sub_C7D6A0(0, 0, 4);
              }
              while ( v51 != v4 );
              v49 += 72 * v51;
            }
            if ( !(v50 % v52) )
              return v48;
            v51 = v52;
            v52 -= v50 % v52;
          }
          v50 = v51;
        }
      }
      v41 = a1;
      v42 = a2;
      do
      {
        v43 = *(_QWORD *)v41;
        ++*(_QWORD *)(v41 + 40);
        v72[0] = 1;
        v67 = v43;
        v44 = *(_QWORD *)(v41 + 8);
        *(_QWORD *)(v41 + 8) = 0;
        v68 = v44;
        v45 = *(_QWORD *)(v41 + 16);
        *(_QWORD *)(v41 + 16) = 0;
        v69 = v45;
        v46 = *(_QWORD *)(v41 + 24);
        *(_QWORD *)(v41 + 24) = 0;
        v70 = v46;
        v71 = *(_QWORD *)(v41 + 32);
        v47 = *(_QWORD *)(v41 + 48);
        *(_QWORD *)(v41 + 48) = 0;
        v72[1] = v47;
        LODWORD(v47) = *(_DWORD *)(v41 + 56);
        *(_DWORD *)(v41 + 56) = 0;
        v73 = v47;
        LODWORD(v47) = *(_DWORD *)(v41 + 60);
        *(_DWORD *)(v41 + 60) = 0;
        v74 = v47;
        LODWORD(v47) = *(_DWORD *)(v41 + 64);
        *(_DWORD *)(v41 + 64) = 0;
        v75 = v47;
        sub_2641BF0(v41, v42);
        sub_2641BF0(v42, (__int64)&v67);
        sub_2342640((__int64)v72);
        if ( v68 )
          j_j___libc_free_0(v68);
        v41 += 72;
        v42 += 72;
      }
      while ( a2 != v41 );
      return a2;
    }
  }
  return result;
}
