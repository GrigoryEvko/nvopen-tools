// Function: sub_659DF0
// Address: 0x659df0
//
__int64 __fastcall sub_659DF0(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  char v4; // al
  __int64 *v5; // rdi
  __int64 *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 result; // rax
  unsigned __int64 v12; // r9
  _QWORD *v13; // r10
  int v14; // esi
  unsigned int v15; // edx
  char *v16; // r8
  __int64 v17; // rax
  int v18; // edx
  unsigned int v19; // edx
  int v20; // eax
  __int64 *v21; // rax
  unsigned int v22; // r8d
  __int64 **v23; // r10
  __int64 *v24; // rcx
  unsigned __int64 v25; // r9
  __int64 *v26; // rsi
  __int64 *v27; // rsi
  __int64 v28; // r11
  unsigned __int64 v29; // rdi
  unsigned __int64 i; // rdx
  unsigned int v31; // edx
  unsigned __int64 *v32; // rax
  int v33; // edx
  unsigned int v34; // edx
  int v35; // eax
  __int64 *v36; // rax
  __int64 *v37; // rsi
  __int64 *v38; // rsi
  unsigned __int64 v39; // rdi
  unsigned __int64 j; // rdx
  unsigned int v41; // edx
  unsigned __int64 *v42; // rax
  unsigned __int64 v43; // [rsp+8h] [rbp-268h]
  unsigned __int64 v44; // [rsp+8h] [rbp-268h]
  __int64 **v45; // [rsp+10h] [rbp-260h]
  __int64 **v46; // [rsp+10h] [rbp-260h]
  unsigned int v47; // [rsp+18h] [rbp-258h]
  unsigned __int64 v48; // [rsp+18h] [rbp-258h]
  unsigned int v49; // [rsp+18h] [rbp-258h]
  unsigned int v50; // [rsp+20h] [rbp-250h]
  __int64 *v51; // [rsp+20h] [rbp-250h]
  unsigned int v52; // [rsp+20h] [rbp-250h]
  unsigned int v53; // [rsp+28h] [rbp-248h]
  unsigned int v54; // [rsp+28h] [rbp-248h]
  unsigned int v55; // [rsp+2Ch] [rbp-244h]
  __int64 v56; // [rsp+30h] [rbp-240h]
  __int64 v57; // [rsp+38h] [rbp-238h]
  __int64 *v58; // [rsp+40h] [rbp-230h]
  unsigned int v59; // [rsp+48h] [rbp-228h]
  unsigned int v60; // [rsp+4Ch] [rbp-224h]
  unsigned int v61; // [rsp+50h] [rbp-220h] BYREF
  int v62; // [rsp+54h] [rbp-21Ch] BYREF
  __int64 v63; // [rsp+58h] [rbp-218h] BYREF
  _QWORD v64[66]; // [rsp+60h] [rbp-210h] BYREF

  v59 = 0;
  v60 = dword_4F04C3C;
  v2 = *(_QWORD *)(*a1 + 88LL);
  v57 = v2;
  v58 = (__int64 *)(v2 + 128);
  if ( (*(_BYTE *)(v2 - 8) & 1) != 0 )
  {
    sub_7296C0(&v61);
    v59 = 1;
  }
  sub_7BC000(a1[46]);
  sub_7BE280(25, 1948, 0, 0);
  ++*(_BYTE *)(qword_4F061C8 + 75LL);
  do
  {
    while ( 1 )
    {
      v62 = 0;
      if ( word_4F06418[0] != 1 )
        break;
      qmemcpy(v64, a1, 0x1D8u);
      BYTE3(v64[16]) &= ~0x10u;
      v64[3] = *(_QWORD *)&dword_4F063F8;
      v64[6] = *(_QWORD *)&dword_4F063F8;
      v3 = v64[38];
      if ( !v64[38] )
      {
        v3 = sub_72B6D0(&dword_4F063F8, 0);
        v64[38] = v3;
      }
      v64[34] = v3;
      v64[35] = v3;
      v64[36] = v3;
      v64[39] = 0;
      BYTE4(v64[33]) = 0;
      v4 = *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4);
      if ( ((v4 - 15) & 0xFD) != 0 && v4 != 2 )
        BYTE5(v64[33]) = 2;
      v5 = &qword_4D04A00;
      v6 = v64;
      v62 = 1;
      dword_4F04C3C = 1;
      sub_6582F0((__m128i *)&qword_4D04A00, (__int64)v64, 0x803u, &v62, &v63, 0);
      dword_4F04C3C = v60;
      if ( !v64[0] || *(_BYTE *)(v64[0] + 80LL) != 7 )
        goto LABEL_11;
      v56 = sub_727640();
      v12 = *(_QWORD *)(v64[0] + 88LL);
      *(_QWORD *)(v12 + 168) = *(_QWORD *)(v12 + 168) & 0xFEFFFFFFFFFEFFFFLL | 0x100000000010000LL;
      *(_QWORD *)(v12 + 128) = v57;
      v13 = qword_4D03BF0;
      v14 = *((_DWORD *)qword_4D03BF0 + 2);
      v8 = *qword_4D03BF0;
      v15 = v14 & (v12 >> 3);
      v5 = (__int64 *)(16LL * v15);
      v16 = (char *)v5 + *qword_4D03BF0;
      if ( *(_QWORD *)v16 )
      {
        do
        {
          v15 = v14 & (v15 + 1);
          v17 = v8 + 16LL * v15;
        }
        while ( *(_QWORD *)v17 );
        v18 = *((_DWORD *)v16 + 2);
        *(_QWORD *)v17 = *(_QWORD *)v16;
        *(_DWORD *)(v17 + 8) = v18;
        *(_QWORD *)v16 = 0;
        v5 = (__int64 *)((char *)v5 + *v13);
        *v5 = v12;
        *((_DWORD *)v5 + 2) = 1;
        v19 = *((_DWORD *)v13 + 2);
        v20 = *((_DWORD *)v13 + 3) + 1;
        *((_DWORD *)v13 + 3) = v20;
        if ( 2 * v20 <= v19 )
          goto LABEL_21;
        v43 = v12;
        v45 = (__int64 **)v13;
        v50 = 2 * v19 + 2;
        v53 = v19;
        v47 = 2 * v19 + 1;
        v55 = v19 + 1;
        v21 = (__int64 *)sub_823970(16LL * v50);
        v22 = v47;
        v23 = v45;
        v24 = v21;
        v25 = v43;
        if ( v50 )
        {
          v26 = &v21[2 * v47 + 2];
          do
          {
            if ( v21 )
              *v21 = 0;
            v21 += 2;
          }
          while ( v26 != v21 );
        }
        v51 = *v45;
        if ( v55 )
        {
          v27 = *v45;
          v28 = (__int64)&v51[2 * v53 + 2];
          do
          {
            while ( 1 )
            {
              v29 = *v27;
              if ( *v27 )
                break;
              v27 += 2;
              if ( (__int64 *)v28 == v27 )
                goto LABEL_35;
            }
            for ( i = v29 >> 3; ; LODWORD(i) = v31 + 1 )
            {
              v31 = v47 & i;
              v32 = (unsigned __int64 *)&v24[2 * v31];
              if ( !*v32 )
                break;
            }
            *v32 = v29;
            v33 = *((_DWORD *)v27 + 2);
            v27 += 2;
            *((_DWORD *)v32 + 2) = v33;
          }
          while ( (__int64 *)v28 != v27 );
        }
      }
      else
      {
        *(_QWORD *)v16 = v12;
        *((_DWORD *)v16 + 2) = 1;
        v34 = *((_DWORD *)v13 + 2);
        v35 = *((_DWORD *)v13 + 3) + 1;
        *((_DWORD *)v13 + 3) = v35;
        if ( 2 * v35 <= v34 )
          goto LABEL_21;
        v44 = v12;
        v46 = (__int64 **)v13;
        v52 = 2 * v34 + 2;
        v54 = v34;
        v49 = 2 * v34 + 1;
        v55 = v34 + 1;
        v36 = (__int64 *)sub_823970(16LL * v52);
        v22 = v49;
        v23 = v46;
        v24 = v36;
        v25 = v44;
        if ( v52 )
        {
          v37 = &v36[2 * v49 + 2];
          do
          {
            if ( v36 )
              *v36 = 0;
            v36 += 2;
          }
          while ( v37 != v36 );
        }
        v51 = *v46;
        if ( v55 )
        {
          v38 = *v46;
          do
          {
            v39 = *v38;
            if ( *v38 )
            {
              for ( j = v39 >> 3; ; LODWORD(j) = v41 + 1 )
              {
                v41 = v49 & j;
                v42 = (unsigned __int64 *)&v24[2 * v41];
                if ( !*v42 )
                  break;
              }
              *v42 = v39;
              *((_DWORD *)v42 + 2) = *((_DWORD *)v38 + 2);
            }
            v38 += 2;
          }
          while ( v38 != &v51[2 * v54 + 2] );
        }
      }
LABEL_35:
      *v23 = v24;
      *((_DWORD *)v23 + 2) = v22;
      v5 = v51;
      v48 = v25;
      sub_823A00(v51, 16LL * v55);
      v12 = v48;
LABEL_21:
      v6 = v58;
      *(_BYTE *)(v56 + 8) = 7;
      *(_QWORD *)(v56 + 16) = v12;
      v58 = (__int64 *)v56;
      *v6 = v56;
      v7 = dword_4F06650[0];
      *(_DWORD *)(v64[0] + 56LL) = dword_4F06650[0];
LABEL_11:
      sub_7B8B50(v5, v6, v7, v8);
      if ( !(unsigned int)sub_7BE800(67) )
        goto LABEL_14;
    }
    sub_6851D0(40);
  }
  while ( (unsigned int)sub_7BE800(67) );
LABEL_14:
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  sub_7BE280(26, 17, 0, 0);
  while ( word_4F06418[0] != 9 )
    sub_7B8B50(26, 17, v9, v10);
  sub_7B8B50(26, 17, v9, v10);
  sub_7AEB40(a1[46]);
  result = v59;
  a1[46] = 0;
  if ( v59 )
    return sub_729730(v61);
  return result;
}
