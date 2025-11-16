// Function: sub_25FED10
// Address: 0x25fed10
//
__int64 *__fastcall sub_25FED10(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 *result; // rax
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // r12
  int v8; // edx
  __int64 v9; // rdi
  int v10; // edx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // r15
  int v20; // edx
  __int64 v21; // rdi
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 *v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rdi
  __int64 v31; // rdi
  char *v32; // [rsp+0h] [rbp-F0h]
  __int64 v33; // [rsp+8h] [rbp-E8h]
  signed __int64 v34; // [rsp+10h] [rbp-E0h]
  signed __int64 v35; // [rsp+18h] [rbp-D8h]
  int v36; // [rsp+20h] [rbp-D0h] BYREF
  int v37; // [rsp+24h] [rbp-CCh]
  __int64 v38; // [rsp+28h] [rbp-C8h]
  __int64 v39; // [rsp+30h] [rbp-C0h]
  __int64 v40; // [rsp+38h] [rbp-B8h]
  __int64 v41; // [rsp+40h] [rbp-B0h]
  int v42; // [rsp+48h] [rbp-A8h]
  int v43; // [rsp+4Ch] [rbp-A4h]
  unsigned int v44; // [rsp+50h] [rbp-A0h]
  __int64 v45; // [rsp+58h] [rbp-98h]
  __int64 v46; // [rsp+60h] [rbp-90h]
  int v47; // [rsp+68h] [rbp-88h]
  int v48; // [rsp+6Ch] [rbp-84h]
  unsigned int v49; // [rsp+70h] [rbp-80h]
  __int64 v50; // [rsp+78h] [rbp-78h]
  __int64 v51; // [rsp+80h] [rbp-70h]
  int v52; // [rsp+88h] [rbp-68h]
  int v53; // [rsp+8Ch] [rbp-64h]
  unsigned int v54; // [rsp+90h] [rbp-60h]
  __int64 v55; // [rsp+98h] [rbp-58h]
  __int64 v56; // [rsp+A0h] [rbp-50h]
  int v57; // [rsp+A8h] [rbp-48h]
  int v58; // [rsp+ACh] [rbp-44h]
  unsigned int v59; // [rsp+B0h] [rbp-40h]

  result = a3;
  v33 = (__int64)a1;
  if ( a1 != a2 )
  {
    result = a1;
    if ( a2 != a3 )
    {
      v32 = (char *)a1 + (char *)a3 - (char *)a2;
      v4 = 0x86BCA1AF286BCA1BLL * (a2 - a1);
      v34 = 0x86BCA1AF286BCA1BLL * (a3 - a1);
      if ( v4 != v34 - v4 )
      {
        while ( 1 )
        {
          v35 = v34 - v4;
          if ( v4 >= v34 - v4 )
          {
            v17 = v33 + 152 * v34;
            v33 = v17 - 152 * v35;
            if ( v4 > 0 )
            {
              v18 = v17 - 152 * v35 - 152;
              v19 = 0;
              do
              {
                v20 = *(_DWORD *)v18;
                ++*(_QWORD *)(v18 + 24);
                v21 = v18;
                ++v19;
                ++*(_QWORD *)(v18 + 56);
                v17 -= 152;
                v36 = v20;
                v22 = *(_DWORD *)(v18 + 4);
                v40 = 1;
                v37 = v22;
                v23 = *(_QWORD *)(v18 + 8);
                v45 = 1;
                v38 = v23;
                v39 = *(_QWORD *)(v18 + 16);
                v24 = *(_QWORD *)(v18 + 32);
                *(_QWORD *)(v18 + 32) = 0;
                v41 = v24;
                LODWORD(v24) = *(_DWORD *)(v18 + 40);
                *(_DWORD *)(v18 + 40) = 0;
                v42 = v24;
                LODWORD(v24) = *(_DWORD *)(v18 + 44);
                *(_DWORD *)(v18 + 44) = 0;
                v43 = v24;
                LODWORD(v24) = *(_DWORD *)(v18 + 48);
                *(_DWORD *)(v18 + 48) = 0;
                v44 = v24;
                v25 = *(_QWORD *)(v18 + 64);
                *(_QWORD *)(v18 + 64) = 0;
                v46 = v25;
                LODWORD(v25) = *(_DWORD *)(v18 + 72);
                *(_DWORD *)(v18 + 72) = 0;
                v47 = v25;
                LODWORD(v25) = *(_DWORD *)(v18 + 76);
                ++*(_QWORD *)(v18 + 88);
                v48 = v25;
                LODWORD(v25) = *(_DWORD *)(v18 + 80);
                ++*(_QWORD *)(v18 + 120);
                v18 -= 152;
                v49 = v25;
                v26 = *(_QWORD *)(v18 + 248);
                *(_DWORD *)(v18 + 228) = 0;
                v51 = v26;
                LODWORD(v26) = *(_DWORD *)(v18 + 256);
                *(_DWORD *)(v18 + 232) = 0;
                v52 = v26;
                LODWORD(v26) = *(_DWORD *)(v18 + 260);
                *(_QWORD *)(v18 + 248) = 0;
                v53 = v26;
                LODWORD(v26) = *(_DWORD *)(v18 + 264);
                *(_DWORD *)(v18 + 256) = 0;
                v54 = v26;
                v27 = *(_QWORD *)(v18 + 280);
                *(_DWORD *)(v18 + 260) = 0;
                v56 = v27;
                LODWORD(v27) = *(_DWORD *)(v18 + 288);
                *(_DWORD *)(v18 + 264) = 0;
                v57 = v27;
                LODWORD(v27) = *(_DWORD *)(v18 + 292);
                *(_QWORD *)(v18 + 280) = 0;
                v58 = v27;
                *(_DWORD *)(v18 + 288) = 0;
                v50 = 1;
                v55 = 1;
                *(_DWORD *)(v18 + 292) = 0;
                LODWORD(v27) = *(_DWORD *)(v18 + 296);
                *(_DWORD *)(v18 + 296) = 0;
                v59 = v27;
                sub_25F6310(v21, v17);
                sub_25F6310(v17, (__int64)&v36);
                sub_C7D6A0(v56, 8LL * v59, 4);
                sub_C7D6A0(v51, 8LL * v54, 4);
                sub_C7D6A0(v46, 16LL * v49, 8);
                sub_C7D6A0(v41, 16LL * v44, 8);
              }
              while ( v4 != v19 );
              v33 += -152 * v4;
            }
            v4 = v34 % v35;
            if ( !(v34 % v35) )
              return (__int64 *)v32;
          }
          else
          {
            v5 = v33;
            if ( v34 - v4 > 0 )
            {
              v6 = v33 + 152 * v4;
              v7 = 0;
              do
              {
                v8 = *(_DWORD *)v5;
                ++*(_QWORD *)(v5 + 24);
                v9 = v5;
                ++*(_QWORD *)(v5 + 56);
                ++v7;
                v36 = v8;
                v10 = *(_DWORD *)(v5 + 4);
                v40 = 1;
                v37 = v10;
                v11 = *(_QWORD *)(v5 + 8);
                v45 = 1;
                v38 = v11;
                v39 = *(_QWORD *)(v5 + 16);
                v12 = *(_QWORD *)(v5 + 32);
                *(_QWORD *)(v5 + 32) = 0;
                v41 = v12;
                LODWORD(v12) = *(_DWORD *)(v5 + 40);
                *(_DWORD *)(v5 + 40) = 0;
                v42 = v12;
                LODWORD(v12) = *(_DWORD *)(v5 + 44);
                *(_DWORD *)(v5 + 44) = 0;
                v43 = v12;
                LODWORD(v12) = *(_DWORD *)(v5 + 48);
                *(_DWORD *)(v5 + 48) = 0;
                v44 = v12;
                v13 = *(_QWORD *)(v5 + 64);
                *(_QWORD *)(v5 + 64) = 0;
                v46 = v13;
                LODWORD(v13) = *(_DWORD *)(v5 + 72);
                *(_DWORD *)(v5 + 72) = 0;
                v47 = v13;
                LODWORD(v13) = *(_DWORD *)(v5 + 76);
                ++*(_QWORD *)(v5 + 88);
                v48 = v13;
                LODWORD(v13) = *(_DWORD *)(v5 + 80);
                ++*(_QWORD *)(v5 + 120);
                v5 += 152;
                v49 = v13;
                v14 = *(_QWORD *)(v5 - 56);
                *(_DWORD *)(v5 - 76) = 0;
                v51 = v14;
                LODWORD(v14) = *(_DWORD *)(v5 - 48);
                *(_DWORD *)(v5 - 72) = 0;
                v52 = v14;
                LODWORD(v14) = *(_DWORD *)(v5 - 44);
                *(_QWORD *)(v5 - 56) = 0;
                v53 = v14;
                LODWORD(v14) = *(_DWORD *)(v5 - 40);
                *(_DWORD *)(v5 - 48) = 0;
                v54 = v14;
                v15 = *(_QWORD *)(v5 - 24);
                *(_DWORD *)(v5 - 44) = 0;
                v56 = v15;
                LODWORD(v15) = *(_DWORD *)(v5 - 16);
                *(_DWORD *)(v5 - 40) = 0;
                v57 = v15;
                LODWORD(v15) = *(_DWORD *)(v5 - 12);
                *(_QWORD *)(v5 - 24) = 0;
                v58 = v15;
                *(_DWORD *)(v5 - 16) = 0;
                v50 = 1;
                v55 = 1;
                *(_DWORD *)(v5 - 12) = 0;
                LODWORD(v15) = *(_DWORD *)(v5 - 8);
                *(_DWORD *)(v5 - 8) = 0;
                v59 = v15;
                sub_25F6310(v9, v6);
                v16 = v6;
                v6 += 152;
                sub_25F6310(v16, (__int64)&v36);
                sub_C7D6A0(v56, 8LL * v59, 4);
                sub_C7D6A0(v51, 8LL * v54, 4);
                sub_C7D6A0(v46, 16LL * v49, 8);
                sub_C7D6A0(v41, 16LL * v44, 8);
              }
              while ( v35 != v7 );
              v33 += 152 * v35;
            }
            if ( !(v34 % v4) )
              return (__int64 *)v32;
            v35 = v4;
            v4 -= v34 % v4;
          }
          v34 = v35;
        }
      }
      v28 = a1;
      v29 = (__int64)a2;
      do
      {
        sub_25FE910((__int64)&v36, v28);
        v30 = (__int64)v28;
        v28 += 19;
        sub_25F6310(v30, v29);
        v31 = v29;
        v29 += 152;
        sub_25F6310(v31, (__int64)&v36);
        sub_C7D6A0(v56, 8LL * v59, 4);
        sub_C7D6A0(v51, 8LL * v54, 4);
        sub_C7D6A0(v46, 16LL * v49, 8);
        sub_C7D6A0(v41, 16LL * v44, 8);
      }
      while ( a2 != v28 );
      return a2;
    }
  }
  return result;
}
