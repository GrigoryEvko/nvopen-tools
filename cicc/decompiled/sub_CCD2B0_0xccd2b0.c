// Function: sub_CCD2B0
// Address: 0xccd2b0
//
unsigned __int64 __fastcall sub_CCD2B0(__int64 a1, int a2, int a3, int a4)
{
  int v4; // eax
  int v7; // r13d
  unsigned __int8 *v8; // rbx
  __int64 v9; // r12
  int v10; // r9d
  unsigned __int8 *v11; // r10
  int v12; // eax
  int v13; // edi
  int v14; // edx
  int v15; // esi
  int v16; // r9d
  int v17; // edx
  int v18; // ecx
  int v19; // esi
  int v20; // eax
  int v21; // edi
  int v22; // eax
  int v23; // ecx
  int v24; // eax
  int v25; // edx
  int v26; // esi
  int v27; // edi
  int v28; // eax
  int v29; // esi
  int v30; // ecx
  int v31; // eax
  int v32; // edx
  int v33; // ecx
  unsigned int v34; // edx
  int v35; // edx
  int v36; // esi
  int v37; // edx
  int v38; // esi
  int v39; // esi
  int v40; // edx
  int v41; // edx
  int v42; // edx
  int v43; // r10d
  int v44; // edx
  int v45; // esi
  int v46; // esi
  int v47; // edi
  int v48; // ecx
  unsigned int v49; // edx
  unsigned int v50; // edx
  __int64 v51; // rax

  v4 = a2 + 7;
  if ( a2 >= 0 )
    v4 = a2;
  v7 = v4 >> 3;
  v8 = (unsigned __int8 *)(a1 + (int)(v4 & 0xFFFFFFF8));
  LODWORD(v9) = -(v4 >> 3);
  if ( a2 <= 31 )
  {
    v12 = 718793509;
    v10 = -1789642873;
  }
  else
  {
    v10 = -1789642873;
    v11 = &v8[-8 * (v4 >> 3)];
    v12 = 718793509;
    do
    {
      v13 = 5 * v10 + 2071795100;
      v14 = __ROL4__(v10 * *(_DWORD *)v11, 11);
      v15 = v10 * __ROL4__(v12 * *((_DWORD *)v11 + 1), 11);
      v16 = 5 * v12 + 1808688022;
      v17 = a4 + (a3 ^ (v12 * v14));
      v18 = 3 * (v17 + (v15 ^ __ROR4__(a4, 15))) + 944331445;
      v19 = v18 + ((3 * v17 + 1390208809) ^ (v16 * __ROL4__(v13 * *((_DWORD *)v11 + 2), 11)));
      v20 = v13 * __ROL4__(v16 * *((_DWORD *)v11 + 3), 11);
      v21 = 5 * v13 + 2071795100;
      v22 = (v20 ^ __ROR4__(v18, 15)) + v19;
      v23 = 5 * v16 + 1808688022;
      v24 = 3 * v22 + 944331445;
      v25 = v24 + ((3 * v19 + 1390208809) ^ (v23 * __ROL4__(v21 * *((_DWORD *)v11 + 4), 11)));
      v26 = v21 * __ROL4__(v23 * *((_DWORD *)v11 + 5), 11);
      v27 = 5 * v21 + 2071795100;
      v10 = 5 * v27 + 2071795100;
      v28 = v26 ^ __ROR4__(v24, 15);
      v29 = 5 * v23 + 1808688022;
      v30 = v27 * *((_DWORD *)v11 + 6);
      v31 = 3 * (v25 + v28) + 944331445;
      LODWORD(v9) = v9 + 4;
      v11 += 32;
      v32 = ((3 * v25 + 1390208809) ^ (v29 * __ROL4__(v30, 11))) + v31;
      a3 = 3 * v32 + 1390208809;
      v33 = ((v27 * __ROL4__(v29 * *((_DWORD *)v11 - 1), 11)) ^ __ROR4__(v31, 15)) + v32;
      v12 = 5 * v29 + 1808688022;
      a4 = 3 * v33 + 944331445;
    }
    while ( (int)v9 < -3 );
    v34 = (v7 - 4) & 0xFFFFFFFC;
    if ( a2 <= 31 )
      v34 = 0;
    LODWORD(v9) = v34 - v7 + 4;
  }
  if ( (_DWORD)v9 )
  {
    v9 = (int)v9;
    do
    {
      v35 = *(_DWORD *)&v8[8 * v9];
      v36 = *(_DWORD *)&v8[8 * v9++ + 4];
      v37 = v12 * __ROL4__(v10 * v35, 11);
      v38 = __ROL4__(v12 * v36, 11);
      v12 = 5 * v12 + 1808688022;
      v39 = v10 * v38;
      v10 = 5 * v10 + 2071795100;
      v40 = a4 + (a3 ^ v37);
      a3 = 3 * v40 + 1390208809;
      a4 = 3 * ((v39 ^ __ROR4__(a4, 15)) + v40) + 944331445;
    }
    while ( (_DWORD)v9 );
  }
  switch ( a2 & 7 )
  {
    case 0:
      goto LABEL_20;
    case 1:
      v43 = 0;
      v46 = 0;
      goto LABEL_19;
    case 2:
      v43 = 0;
      v45 = 0;
      goto LABEL_18;
    case 3:
      v43 = 0;
      v44 = 0;
      goto LABEL_17;
    case 4:
      v43 = 0;
      goto LABEL_16;
    case 5:
      v42 = 0;
      goto LABEL_15;
    case 6:
      v41 = 0;
      goto LABEL_14;
    case 7:
      v41 = v8[6] << 16;
LABEL_14:
      v42 = (v8[5] << 8) ^ v41;
LABEL_15:
      v43 = v10 * __ROL4__(v12 * (v42 ^ v8[4]), 11);
LABEL_16:
      v44 = v8[3] << 24;
LABEL_17:
      v45 = v44 ^ (v8[2] << 16);
LABEL_18:
      v46 = (v8[1] << 8) ^ v45;
LABEL_19:
      v47 = (a3 ^ (__ROL4__(v10 * (v46 ^ *v8), 11) * v12)) + a4;
      a3 = 3 * v47 + 1390208809;
      a4 = 3 * ((v43 ^ __ROR4__(a4, 15)) + v47) + 944331445;
LABEL_20:
      v48 = a2 ^ a4;
      v49 = -2048144789 * ((v48 + v48 + a3) ^ ((unsigned int)(v48 + v48 + a3) >> 16));
      v50 = ((-1028477387 * ((v49 >> 13) ^ v49)) >> 16) ^ (-1028477387 * ((v49 >> 13) ^ v49));
      v51 = v50
          + (((-1028477387
             * (((-2048144789 * (((unsigned int)(v48 + a3) >> 16) ^ (v48 + a3))) >> 13)
              ^ (-2048144789 * (((unsigned int)(v48 + a3) >> 16) ^ (v48 + a3))))) >> 16)
           ^ (-1028477387
            * (((-2048144789 * (((unsigned int)(v48 + a3) >> 16) ^ (v48 + a3))) >> 13)
             ^ (-2048144789 * (((unsigned int)(v48 + a3) >> 16) ^ (v48 + a3))))));
      return ((unsigned __int64)((unsigned int)v51 + v50) << 32) | v51;
  }
}
