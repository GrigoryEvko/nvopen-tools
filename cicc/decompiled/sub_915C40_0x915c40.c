// Function: sub_915C40
// Address: 0x915c40
//
__int64 __fastcall sub_915C40(__int64 *a1, const char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r9
  unsigned int v11; // esi
  __int64 v12; // rdi
  unsigned int v13; // r10d
  _QWORD *v14; // rax
  __int64 v15; // rcx
  const __m128i *v16; // r15
  const __m128i **v17; // r12
  _QWORD *v19; // r12
  int v20; // eax
  int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  size_t v27; // rax
  __int64 v28; // rcx
  size_t v29; // r8
  _QWORD *v30; // rdx
  char v31; // r14
  char v32; // r14
  char *v33; // r13
  char *v34; // r8
  unsigned __int64 v35; // rdx
  char *v36; // rax
  char v37; // cl
  __int64 v38; // rax
  char *v39; // rax
  char v40; // cl
  _BOOL4 v41; // r14d
  __int64 v42; // rax
  int v43; // esi
  int v44; // esi
  __int64 v45; // r9
  unsigned int v46; // ecx
  __int64 v47; // rdi
  int v48; // r15d
  _QWORD *v49; // r11
  int v50; // ecx
  int v51; // ecx
  __int64 v52; // rdi
  _QWORD *v53; // r9
  unsigned int v54; // r15d
  int v55; // r11d
  __int64 v56; // rsi
  __int64 v57; // rax
  _QWORD *v58; // rdi
  __int64 v59; // rax
  unsigned int n; // [rsp+0h] [rbp-C0h]
  size_t na; // [rsp+0h] [rbp-C0h]
  __int64 v62; // [rsp+8h] [rbp-B8h]
  int v63; // [rsp+8h] [rbp-B8h]
  int v64; // [rsp+8h] [rbp-B8h]
  int v65; // [rsp+10h] [rbp-B0h]
  __int64 v66; // [rsp+10h] [rbp-B0h]
  __int64 v67; // [rsp+10h] [rbp-B0h]
  __int64 v68; // [rsp+10h] [rbp-B0h]
  char *sa; // [rsp+18h] [rbp-A8h]
  _QWORD v71[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v72[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v73[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v74; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v75[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v76; // [rsp+80h] [rbp-40h]

  v9 = (__int64)(a1 + 47);
  v11 = *((_DWORD *)a1 + 100);
  if ( !v11 )
  {
    ++a1[47];
    goto LABEL_65;
  }
  v12 = a1[48];
  v13 = (v11 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
  v14 = (_QWORD *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( a6 != *v14 )
  {
    v65 = 1;
    v19 = 0;
    while ( v15 != -4096 )
    {
      if ( !v19 && v15 == -8192 )
        v19 = v14;
      v13 = (v11 - 1) & (v65 + v13);
      v14 = (_QWORD *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( a6 == *v14 )
        goto LABEL_3;
      ++v65;
    }
    if ( !v19 )
      v19 = v14;
    v20 = *((_DWORD *)a1 + 98);
    ++a1[47];
    v21 = v20 + 1;
    if ( 4 * v21 < 3 * v11 )
    {
      if ( v11 - *((_DWORD *)a1 + 99) - v21 > v11 >> 3 )
      {
LABEL_12:
        *((_DWORD *)a1 + 98) = v21;
        if ( *v19 != -4096 )
          --*((_DWORD *)a1 + 99);
        *v19 = a6;
        v17 = (const __m128i **)(v19 + 1);
        *v17 = 0;
LABEL_15:
        v22 = *(_DWORD *)(a3 + 8);
        v23 = *a1;
        BYTE4(v73[0]) = 1;
        n = a5;
        v62 = a4;
        LODWORD(v73[0]) = v22 >> 8;
        v66 = v23;
        v76 = 257;
        v24 = sub_BD2C40(88, unk_3F0FAE8);
        v16 = (const __m128i *)v24;
        if ( v24 )
          sub_B30000(v24, v66, v62, 0, n, 0, v75, 0, 0, v73[0], 0);
        LODWORD(v25) = sub_91CB50(a6);
        v26 = 0;
        if ( (_DWORD)v25 )
        {
          _BitScanReverse64((unsigned __int64 *)&v25, (unsigned int)v25);
          LOBYTE(v26) = 63 - (v25 ^ 0x3F);
          BYTE1(v26) = 1;
        }
        sub_B2F740(v16, v26);
        if ( (*(_BYTE *)(a6 + 156) & 1) == 0 && (*(_BYTE *)(a6 + 176) & 0x20) != 0 && *(_BYTE *)(a6 + 136) != 1 )
          sub_916690(a1, a6, v16);
        if ( (*(_BYTE *)(a6 + 157) & 1) != 0 )
          sub_913680(a1, v16, (__int64)"managed", 1u);
        if ( !a2 )
        {
LABEL_31:
          if ( unk_4D046B4 && *(char *)(a6 + 173) >= 0 )
            sub_943430(a1[46], v16, a6, 0);
          if ( (a1[45] & 1) == 0 )
            goto LABEL_35;
          v31 = sub_91C2A0(*(_QWORD *)(a6 + 120));
          if ( !v31 )
          {
            v32 = sub_91C2D0(*(_QWORD *)(a6 + 120));
            if ( !v32 )
              goto LABEL_35;
            sub_913680(a1, v16, (__int64)"surface", 1u);
            v33 = (char *)a1[65];
            v34 = (char *)(a1 + 64);
            if ( v33 )
            {
              while ( 1 )
              {
                v35 = *((_QWORD *)v33 + 4);
                v36 = (char *)*((_QWORD *)v33 + 3);
                v37 = 0;
                if ( (unsigned __int64)v16 < v35 )
                {
                  v36 = (char *)*((_QWORD *)v33 + 2);
                  v37 = v32;
                }
                if ( !v36 )
                  break;
                v33 = v36;
              }
              if ( v37 )
              {
                if ( (char *)a1[66] == v33 )
                  goto LABEL_61;
LABEL_47:
                v38 = sub_220EF80(v33);
                v34 = (char *)(a1 + 64);
                if ( (unsigned __int64)v16 > *(_QWORD *)(v38 + 32) )
                {
LABEL_61:
                  v41 = 1;
                  if ( v33 != v34 )
                    v41 = (unsigned __int64)v16 < *((_QWORD *)v33 + 4);
                  goto LABEL_63;
                }
LABEL_35:
                *v17 = v16;
                return (__int64)v16;
              }
LABEL_60:
              if ( v35 < (unsigned __int64)v16 )
                goto LABEL_61;
              goto LABEL_35;
            }
            v33 = (char *)(a1 + 64);
            if ( (char *)a1[66] != v34 )
              goto LABEL_47;
LABEL_89:
            v41 = 1;
LABEL_63:
            sa = v34;
            v42 = sub_22077B0(40);
            *(_QWORD *)(v42 + 32) = v16;
            sub_220F040(v41, v42, v33, sa);
            ++a1[68];
            goto LABEL_35;
          }
          sub_913680(a1, v16, (__int64)"texture", 1u);
          v33 = (char *)a1[65];
          v34 = (char *)(a1 + 64);
          if ( v33 )
          {
            while ( 1 )
            {
              v35 = *((_QWORD *)v33 + 4);
              v39 = (char *)*((_QWORD *)v33 + 3);
              v40 = 0;
              if ( (unsigned __int64)v16 < v35 )
              {
                v39 = (char *)*((_QWORD *)v33 + 2);
                v40 = v31;
              }
              if ( !v39 )
                break;
              v33 = v39;
            }
            if ( !v40 )
              goto LABEL_60;
            if ( (char *)a1[66] == v33 )
              goto LABEL_61;
          }
          else
          {
            v33 = (char *)(a1 + 64);
            if ( (char *)a1[66] == v34 )
              goto LABEL_89;
          }
          v59 = sub_220EF80(v33);
          v34 = (char *)(a1 + 64);
          if ( *(_QWORD *)(v59 + 32) >= (unsigned __int64)v16 )
            goto LABEL_35;
          goto LABEL_61;
        }
        v71[0] = v72;
        v27 = strlen(a2);
        v75[0] = v27;
        v29 = v27;
        if ( v27 > 0xF )
        {
          na = v27;
          v57 = sub_22409D0(v71, v75, 0);
          v29 = na;
          v71[0] = v57;
          v58 = (_QWORD *)v57;
          v72[0] = v75[0];
        }
        else
        {
          if ( v27 == 1 )
          {
            LOBYTE(v72[0]) = *a2;
            v30 = v72;
LABEL_27:
            v71[1] = v27;
            *((_BYTE *)v30 + v27) = 0;
            sub_91B9C0(v73, v71, a6, v28, v29);
            v76 = 260;
            v75[0] = v73;
            sub_BD6B50(v16, v75);
            if ( (__int64 *)v73[0] != &v74 )
              j_j___libc_free_0(v73[0], v74 + 1);
            if ( (_QWORD *)v71[0] != v72 )
              j_j___libc_free_0(v71[0], v72[0] + 1LL);
            goto LABEL_31;
          }
          if ( !v27 )
          {
            v30 = v72;
            goto LABEL_27;
          }
          v58 = v72;
        }
        memcpy(v58, a2, v29);
        v27 = v75[0];
        v30 = (_QWORD *)v71[0];
        goto LABEL_27;
      }
      v64 = a5;
      v68 = a4;
      sub_915A60(v9, v11);
      v50 = *((_DWORD *)a1 + 100);
      if ( v50 )
      {
        v51 = v50 - 1;
        v52 = a1[48];
        v53 = 0;
        v54 = v51 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
        a4 = v68;
        LODWORD(a5) = v64;
        v55 = 1;
        v21 = *((_DWORD *)a1 + 98) + 1;
        v19 = (_QWORD *)(v52 + 16LL * v54);
        v56 = *v19;
        if ( a6 != *v19 )
        {
          while ( v56 != -4096 )
          {
            if ( !v53 && v56 == -8192 )
              v53 = v19;
            v54 = v51 & (v55 + v54);
            v19 = (_QWORD *)(v52 + 16LL * v54);
            v56 = *v19;
            if ( a6 == *v19 )
              goto LABEL_12;
            ++v55;
          }
          if ( v53 )
            v19 = v53;
        }
        goto LABEL_12;
      }
LABEL_103:
      ++*((_DWORD *)a1 + 98);
      BUG();
    }
LABEL_65:
    v63 = a5;
    v67 = a4;
    sub_915A60(v9, 2 * v11);
    v43 = *((_DWORD *)a1 + 100);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = a1[48];
      a4 = v67;
      LODWORD(a5) = v63;
      v46 = v44 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
      v21 = *((_DWORD *)a1 + 98) + 1;
      v19 = (_QWORD *)(v45 + 16LL * v46);
      v47 = *v19;
      if ( a6 != *v19 )
      {
        v48 = 1;
        v49 = 0;
        while ( v47 != -4096 )
        {
          if ( !v49 && v47 == -8192 )
            v49 = v19;
          v46 = v44 & (v48 + v46);
          v19 = (_QWORD *)(v45 + 16LL * v46);
          v47 = *v19;
          if ( a6 == *v19 )
            goto LABEL_12;
          ++v48;
        }
        if ( v49 )
          v19 = v49;
      }
      goto LABEL_12;
    }
    goto LABEL_103;
  }
LABEL_3:
  v16 = (const __m128i *)v14[1];
  v17 = (const __m128i **)(v14 + 1);
  if ( !v16 )
    goto LABEL_15;
  if ( a3 != v16->m128i_i64[1] )
    return sub_AD4C90(v14[1], a3, 0, v15, a5, v9);
  return (__int64)v16;
}
