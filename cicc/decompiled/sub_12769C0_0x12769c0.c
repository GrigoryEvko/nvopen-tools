// Function: sub_12769C0
// Address: 0x12769c0
//
__int64 __fastcall sub_12769C0(__int64 *a1, const char *a2, __int64 a3, int a4, __int64 a5)
{
  __int64 v5; // r9
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rdx
  const __m128i *v14; // r15
  int v16; // r11d
  _QWORD *v17; // r13
  int v18; // eax
  int v19; // edx
  int v20; // r14d
  __int64 v21; // rax
  unsigned int v22; // eax
  size_t v23; // rax
  size_t v24; // r9
  _QWORD *v25; // rdx
  char v26; // r14
  char v27; // r14
  char *v28; // r12
  char *v29; // r8
  unsigned __int64 v30; // rdx
  char *v31; // rax
  char v32; // cl
  _BOOL4 v33; // r14d
  unsigned __int64 v34; // rdx
  char *v35; // rax
  char v36; // cl
  __int64 v37; // rax
  int v38; // eax
  int v39; // edi
  __int64 v40; // rsi
  unsigned int v41; // eax
  __int64 v42; // r8
  int v43; // r10d
  _QWORD *v44; // r9
  int v45; // eax
  int v46; // eax
  __int64 v47; // rdi
  _QWORD *v48; // r8
  unsigned int v49; // r15d
  int v50; // r9d
  __int64 v51; // rsi
  __int64 v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rax
  int n; // [rsp+0h] [rbp-B0h]
  size_t na; // [rsp+0h] [rbp-B0h]
  __int64 v58; // [rsp+8h] [rbp-A8h]
  __int64 v59; // [rsp+10h] [rbp-A0h]
  int v60; // [rsp+10h] [rbp-A0h]
  int v61; // [rsp+10h] [rbp-A0h]
  char *sa; // [rsp+18h] [rbp-98h]
  _QWORD *v64; // [rsp+20h] [rbp-90h] BYREF
  __int16 v65; // [rsp+30h] [rbp-80h]
  _QWORD v66[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v67[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v68[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v69; // [rsp+70h] [rbp-40h] BYREF

  v5 = (__int64)(a1 + 49);
  v9 = *((_DWORD *)a1 + 104);
  if ( !v9 )
  {
    ++a1[49];
    goto LABEL_62;
  }
  v10 = a1[50];
  v11 = (v9 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
  v12 = (_QWORD *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( a5 != *v12 )
  {
    v16 = 1;
    v17 = 0;
    while ( v13 != -4 )
    {
      if ( !v17 && v13 == -8 )
        v17 = v12;
      v11 = (v9 - 1) & (v16 + v11);
      v12 = (_QWORD *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( a5 == *v12 )
        goto LABEL_3;
      ++v16;
    }
    if ( !v17 )
      v17 = v12;
    v18 = *((_DWORD *)a1 + 102);
    ++a1[49];
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v9 )
    {
      if ( v9 - *((_DWORD *)a1 + 103) - v19 > v9 >> 3 )
      {
LABEL_12:
        *((_DWORD *)a1 + 102) = v19;
        if ( *v17 != -4 )
          --*((_DWORD *)a1 + 103);
        *v17 = a5;
        v17[1] = 0;
LABEL_15:
        n = a4;
        v58 = *a1;
        v59 = *(_QWORD *)(a3 + 24);
        v20 = *(_DWORD *)(a3 + 8) >> 8;
        LOWORD(v69) = 257;
        v21 = sub_1648A60(88, 1);
        v14 = (const __m128i *)v21;
        if ( v21 )
          sub_15E51E0(v21, v58, v59, 0, n, 0, (__int64)v68, 0, 0, v20, 0);
        v22 = sub_127C800(a5);
        sub_15E4CC0(v14, v22);
        if ( (*(_BYTE *)(a5 + 156) & 1) == 0 && (*(_BYTE *)(a5 + 176) & 0x20) != 0 && *(_BYTE *)(a5 + 136) != 1 )
          sub_1277390(a1, a5, v14);
        if ( (*(_BYTE *)(a5 + 157) & 1) != 0 )
          sub_1273260(a1, v14, (__int64)"managed", 1u);
        if ( !a2 )
        {
LABEL_29:
          if ( dword_4D046B4 && *(char *)(a5 + 173) >= 0 )
            sub_12A24A0(a1[48], v14, a5, 0);
          if ( (a1[47] & 1) == 0 )
            goto LABEL_33;
          v26 = sub_127BF50(*(_QWORD *)(a5 + 120));
          if ( !v26 )
          {
            v27 = sub_127BF80(*(_QWORD *)(a5 + 120));
            if ( !v27 )
              goto LABEL_33;
            sub_1273260(a1, v14, (__int64)"surface", 1u);
            v28 = (char *)a1[69];
            v29 = (char *)(a1 + 68);
            if ( v28 )
            {
              while ( 1 )
              {
                v30 = *((_QWORD *)v28 + 4);
                v31 = (char *)*((_QWORD *)v28 + 3);
                v32 = 0;
                if ( (unsigned __int64)v14 < v30 )
                {
                  v31 = (char *)*((_QWORD *)v28 + 2);
                  v32 = v27;
                }
                if ( !v31 )
                  break;
                v28 = v31;
              }
              if ( !v32 )
              {
                if ( v30 < (unsigned __int64)v14 )
                  goto LABEL_45;
LABEL_33:
                v17[1] = v14;
                return (__int64)v14;
              }
              if ( (char *)a1[70] == v28 )
              {
LABEL_45:
                v33 = 1;
                if ( v28 == v29 )
                {
LABEL_60:
                  sa = v29;
                  v37 = sub_22077B0(40);
                  *(_QWORD *)(v37 + 32) = v14;
                  sub_220F040(v33, v37, v28, sa);
                  ++a1[72];
                  goto LABEL_33;
                }
LABEL_46:
                v33 = (unsigned __int64)v14 < *((_QWORD *)v28 + 4);
                goto LABEL_60;
              }
LABEL_86:
              v55 = sub_220EF80(v28);
              v29 = (char *)(a1 + 68);
              if ( (unsigned __int64)v14 <= *(_QWORD *)(v55 + 32) )
                goto LABEL_33;
              goto LABEL_45;
            }
            v28 = (char *)(a1 + 68);
            if ( (char *)a1[70] != v29 )
              goto LABEL_86;
LABEL_89:
            v33 = 1;
            goto LABEL_60;
          }
          sub_1273260(a1, v14, (__int64)"texture", 1u);
          v28 = (char *)a1[69];
          v29 = (char *)(a1 + 68);
          if ( v28 )
          {
            while ( 1 )
            {
              v34 = *((_QWORD *)v28 + 4);
              v35 = (char *)*((_QWORD *)v28 + 3);
              v36 = 0;
              if ( (unsigned __int64)v14 < v34 )
              {
                v35 = (char *)*((_QWORD *)v28 + 2);
                v36 = v26;
              }
              if ( !v35 )
                break;
              v28 = v35;
            }
            if ( !v36 )
            {
              if ( (unsigned __int64)v14 <= v34 )
                goto LABEL_33;
LABEL_59:
              v33 = 1;
              if ( v29 == v28 )
                goto LABEL_60;
              goto LABEL_46;
            }
            if ( (char *)a1[70] == v28 )
              goto LABEL_59;
          }
          else
          {
            v28 = (char *)(a1 + 68);
            if ( v29 == (char *)a1[70] )
              goto LABEL_89;
          }
          v54 = sub_220EF80(v28);
          v29 = (char *)(a1 + 68);
          if ( *(_QWORD *)(v54 + 32) >= (unsigned __int64)v14 )
            goto LABEL_33;
          goto LABEL_59;
        }
        v66[0] = v67;
        v23 = strlen(a2);
        v68[0] = v23;
        v24 = v23;
        if ( v23 > 0xF )
        {
          na = v23;
          v52 = sub_22409D0(v66, v68, 0);
          v24 = na;
          v66[0] = v52;
          v53 = (_QWORD *)v52;
          v67[0] = v68[0];
        }
        else
        {
          if ( v23 == 1 )
          {
            LOBYTE(v67[0]) = *a2;
            v25 = v67;
LABEL_25:
            v66[1] = v23;
            *((_BYTE *)v25 + v23) = 0;
            sub_127B670(v68, v66, a5);
            v65 = 260;
            v64 = v68;
            sub_164B780(v14, &v64);
            if ( (__int64 *)v68[0] != &v69 )
              j_j___libc_free_0(v68[0], v69 + 1);
            if ( (_QWORD *)v66[0] != v67 )
              j_j___libc_free_0(v66[0], v67[0] + 1LL);
            goto LABEL_29;
          }
          if ( !v23 )
          {
            v25 = v67;
            goto LABEL_25;
          }
          v53 = v67;
        }
        memcpy(v53, a2, v24);
        v23 = v68[0];
        v25 = (_QWORD *)v66[0];
        goto LABEL_25;
      }
      v61 = a4;
      sub_12755C0(v5, v9);
      v45 = *((_DWORD *)a1 + 104);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = a1[50];
        v48 = 0;
        v49 = v46 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
        v50 = 1;
        v19 = *((_DWORD *)a1 + 102) + 1;
        a4 = v61;
        v17 = (_QWORD *)(v47 + 16LL * v49);
        v51 = *v17;
        if ( a5 != *v17 )
        {
          while ( v51 != -4 )
          {
            if ( !v48 && v51 == -8 )
              v48 = v17;
            v49 = v46 & (v50 + v49);
            v17 = (_QWORD *)(v47 + 16LL * v49);
            v51 = *v17;
            if ( a5 == *v17 )
              goto LABEL_12;
            ++v50;
          }
          if ( v48 )
            v17 = v48;
        }
        goto LABEL_12;
      }
LABEL_104:
      ++*((_DWORD *)a1 + 102);
      BUG();
    }
LABEL_62:
    v60 = a4;
    sub_12755C0(v5, 2 * v9);
    v38 = *((_DWORD *)a1 + 104);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = a1[50];
      v41 = (v38 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
      v19 = *((_DWORD *)a1 + 102) + 1;
      a4 = v60;
      v17 = (_QWORD *)(v40 + 16LL * v41);
      v42 = *v17;
      if ( a5 != *v17 )
      {
        v43 = 1;
        v44 = 0;
        while ( v42 != -4 )
        {
          if ( !v44 && v42 == -8 )
            v44 = v17;
          v41 = v39 & (v43 + v41);
          v17 = (_QWORD *)(v40 + 16LL * v41);
          v42 = *v17;
          if ( a5 == *v17 )
            goto LABEL_12;
          ++v43;
        }
        if ( v44 )
          v17 = v44;
      }
      goto LABEL_12;
    }
    goto LABEL_104;
  }
LABEL_3:
  v14 = (const __m128i *)v12[1];
  if ( !v14 )
  {
    v17 = v12;
    goto LABEL_15;
  }
  if ( a3 != v14->m128i_i64[0] )
    return sub_15A4510(v12[1], a3, 0);
  return (__int64)v14;
}
