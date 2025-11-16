// Function: sub_1B491E0
// Address: 0x1b491e0
//
__int64 __fastcall sub_1B491E0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v9; // r8d
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r14
  char v22; // r8
  unsigned int v23; // edi
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // r9d
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rdi
  __int64 v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r12
  __int64 v39; // rbx
  __int64 v40; // rdx
  _QWORD *v41; // rax
  __int64 v42; // rdi
  unsigned __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  __int64 j; // r13
  unsigned __int64 v52; // rax
  __int64 v53; // rdi
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 v56; // r14
  __int64 v57; // r13
  __int64 v58; // r15
  _QWORD *v59; // r13
  _QWORD *v60; // rax
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rdx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // r15
  __int64 v68; // rbx
  __int64 v69; // r13
  __int64 v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // rdx
  __int64 v73; // [rsp+8h] [rbp-68h]
  __int64 i; // [rsp+10h] [rbp-60h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  __int64 v76; // [rsp+20h] [rbp-50h]
  __int64 v77; // [rsp+28h] [rbp-48h]
  __int64 v78; // [rsp+28h] [rbp-48h]
  __int64 v79[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = 0;
  v10 = *(_QWORD *)(a1 + 40);
  v11 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v12 = (__int64 *)(a1 - 24 * v11);
  v13 = *v12;
  if ( v10 == *(_QWORD *)(*v12 + 40) )
  {
    v15 = *(_QWORD *)(v13 + 8);
    if ( v15 )
    {
      v75 = *(_QWORD *)(v15 + 8);
      if ( !v75 )
      {
        v16 = v13 + 24;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v16 + 8);
          if ( a1 + 24 == v16 )
            break;
          if ( !v16 )
            BUG();
          if ( *(_BYTE *)(v16 - 8) != 78 )
            return 0;
          v17 = *(_QWORD *)(v16 - 48);
          if ( *(_BYTE *)(v17 + 16) || (*(_BYTE *)(v17 + 33) & 0x20) == 0 )
            return 0;
          v18 = *(_DWORD *)(v17 + 36);
          if ( v18 > 0x26 )
          {
            if ( v18 != 116 )
              return 0;
          }
          else if ( v18 <= 0x23 )
          {
            return 0;
          }
        }
        if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
        {
          v75 = *(_QWORD *)(a1 + 24 * (1 - v11));
          if ( v75 )
          {
            v19 = sub_157ED20(v75);
            v20 = *(_QWORD *)(v75 + 48);
            v73 = v19;
            for ( i = v19 + 24; i != v20; v20 = *(_QWORD *)(v20 + 8) )
            {
              if ( !v20 )
                BUG();
              v21 = v20 - 24;
              v22 = *(_BYTE *)(v20 - 1) & 0x40;
              v23 = *(_DWORD *)(v20 - 4) & 0xFFFFFFF;
              if ( v23 )
              {
                v24 = 24LL * *(unsigned int *)(v20 + 32) + 8;
                v25 = 0;
                while ( 1 )
                {
                  v26 = v25;
                  v27 = v21 - 24LL * v23;
                  if ( v22 )
                    v27 = *(_QWORD *)(v20 - 32);
                  if ( v10 == *(_QWORD *)(v27 + v24) )
                    break;
                  ++v25;
                  v24 += 8;
                  if ( v23 == (_DWORD)v25 )
                    goto LABEL_75;
                }
                v28 = 24 * v25;
                if ( v22 )
                {
LABEL_28:
                  v29 = *(_QWORD *)(v20 - 32);
                  goto LABEL_29;
                }
              }
              else
              {
LABEL_75:
                v28 = 0x17FFFFFFE8LL;
                v26 = -1;
                if ( v22 )
                  goto LABEL_28;
              }
              v29 = v21 - 24LL * v23;
LABEL_29:
              v30 = *(_QWORD *)(v29 + v28);
              v31 = v20 - 24;
              if ( *(_BYTE *)(v30 + 16) != 77 )
              {
                sub_15F5350(v31, v26, 0);
LABEL_31:
                v79[0] = *(_QWORD *)(v10 + 8);
                sub_15CDD40(v79);
                v32 = v79[0];
                if ( v79[0] )
                {
                  v33 = sub_1648700(v79[0]);
                  v76 = v10;
                  v38 = v20;
                  v39 = v32;
LABEL_44:
                  v47 = v33[5];
                  v48 = *(_DWORD *)(v38 - 4) & 0xFFFFFFF;
                  if ( v48 == *(_DWORD *)(v38 + 32) )
                  {
                    sub_15F55D0(v21, v47, v34, v35, v36, v37);
                    v48 = *(_DWORD *)(v38 - 4) & 0xFFFFFFF;
                  }
                  v49 = (v48 + 1) & 0xFFFFFFF;
                  v50 = v49 | *(_DWORD *)(v38 - 4) & 0xF0000000;
                  *(_DWORD *)(v38 - 4) = v50;
                  if ( (v50 & 0x40000000) != 0 )
                    v40 = *(_QWORD *)(v38 - 32);
                  else
                    v40 = v21 - 24 * v49;
                  v41 = (_QWORD *)(v40 + 24LL * (unsigned int)(v49 - 1));
                  if ( *v41 )
                  {
                    v42 = v41[1];
                    v43 = v41[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v43 = v42;
                    if ( v42 )
                      *(_QWORD *)(v42 + 16) = *(_QWORD *)(v42 + 16) & 3LL | v43;
                  }
                  *v41 = v30;
                  v44 = *(_QWORD *)(v30 + 8);
                  v41[1] = v44;
                  if ( v44 )
                    *(_QWORD *)(v44 + 16) = (unsigned __int64)(v41 + 1) | *(_QWORD *)(v44 + 16) & 3LL;
                  v41[2] = (v30 + 8) | v41[2] & 3LL;
                  *(_QWORD *)(v30 + 8) = v41;
                  v45 = *(_DWORD *)(v38 - 4) & 0xFFFFFFF;
                  if ( (*(_BYTE *)(v38 - 1) & 0x40) != 0 )
                    v46 = *(_QWORD *)(v38 - 32);
                  else
                    v46 = v21 - 24 * v45;
                  *(_QWORD *)(v46 + 8LL * (unsigned int)(v45 - 1) + 24LL * *(unsigned int *)(v38 + 32) + 8) = v47;
                  while ( 1 )
                  {
                    v39 = *(_QWORD *)(v39 + 8);
                    if ( !v39 )
                      break;
                    v33 = sub_1648700(v39);
                    v35 = *((unsigned __int8 *)v33 + 16);
                    v34 = (unsigned int)(v35 - 25);
                    if ( (unsigned __int8)(v35 - 25) <= 9u )
                      goto LABEL_44;
                  }
                  v20 = v38;
                  v10 = v76;
                }
                continue;
              }
              sub_15F5350(v31, v26, 0);
              if ( v10 != *(_QWORD *)(v30 + 40) )
                goto LABEL_31;
              v67 = 0;
              if ( (*(_DWORD *)(v30 + 20) & 0xFFFFFFF) != 0 )
              {
                v78 = v20;
                v68 = v30;
                v69 = 8LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF);
                do
                {
                  if ( (*(_BYTE *)(v68 + 23) & 0x40) != 0 )
                    v70 = *(_QWORD *)(v68 - 8);
                  else
                    v70 = v68 - 24LL * (*(_DWORD *)(v68 + 20) & 0xFFFFFFF);
                  v72 = *(_QWORD *)(v70 + v67 + 24LL * *(unsigned int *)(v68 + 56) + 8);
                  v71 = 3 * v67;
                  v67 += 8;
                  sub_1704F80(v21, *(_QWORD *)(v70 + v71), v72, v70, v65, v66);
                }
                while ( v69 != v67 );
                v20 = v78;
              }
            }
            v56 = *(_QWORD *)(v10 + 48);
            v77 = sub_157ED20(v10) + 24;
            while ( v77 != v56 )
            {
              v57 = v56;
              v56 = *(_QWORD *)(v56 + 8);
              if ( *(_QWORD *)(v57 - 16) )
              {
                v79[0] = *(_QWORD *)(v75 + 8);
                sub_15CDD40(v79);
                v58 = v79[0];
                if ( v79[0] )
                {
                  v59 = (_QWORD *)(v57 - 24);
                  v60 = sub_1648700(v79[0]);
LABEL_64:
                  v64 = v60[5];
                  if ( v10 != v64 )
                    sub_1704F80((__int64)v59, (__int64)v59, v64, v61, v62, v63);
                  while ( 1 )
                  {
                    v58 = *(_QWORD *)(v58 + 8);
                    if ( !v58 )
                      break;
                    v60 = sub_1648700(v58);
                    if ( (unsigned __int8)(*((_BYTE *)v60 + 16) - 25) <= 9u )
                      goto LABEL_64;
                  }
                }
                else
                {
                  v59 = (_QWORD *)(v57 - 24);
                }
                sub_15F22F0(v59, v73);
              }
            }
          }
        }
        v79[0] = *(_QWORD *)(v10 + 8);
        sub_15CDD40(v79);
        for ( j = v79[0]; v79[0]; j = v79[0] )
        {
          v79[0] = *(_QWORD *)(j + 8);
          sub_15CDD40(v79);
          v53 = sub_1648700(j)[5];
          if ( v75 )
          {
            v52 = sub_157EBA0(v53);
            sub_1648780(v52, v10, v75);
          }
          else
          {
            sub_1AF0970(v53, 0, a2, a3, a4, a5, v54, v55, a8, a9);
          }
        }
        sub_157F980(v10);
        return 1;
      }
    }
  }
  return v9;
}
