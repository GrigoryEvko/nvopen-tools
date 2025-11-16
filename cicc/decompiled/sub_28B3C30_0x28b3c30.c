// Function: sub_28B3C30
// Address: 0x28b3c30
//
__int64 __fastcall sub_28B3C30(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // rbx
  unsigned int v7; // r15d
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // eax
  unsigned __int64 v11; // r13
  __int64 v12; // r14
  int v13; // eax
  __int64 *v14; // rdx
  __int64 v15; // r8
  int v16; // eax
  __int64 v17; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // ebx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned int v29; // ebx
  __int64 v30; // r12
  int v31; // edx
  __int64 v32; // r15
  unsigned int v33; // eax
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r14
  int v38; // r14d
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // edx
  __int64 v43; // r15
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r14
  int v47; // r14d
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned int v51; // edx
  _DWORD *v52; // rax
  int v53; // edx
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  int v58; // r14d
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rdx
  unsigned int v62; // eax
  __int64 v63; // [rsp+8h] [rbp-78h]
  int v64; // [rsp+14h] [rbp-6Ch]
  __int64 v65; // [rsp+18h] [rbp-68h]
  __int64 v66; // [rsp+18h] [rbp-68h]
  _QWORD *v67; // [rsp+18h] [rbp-68h]
  __int64 v68; // [rsp+20h] [rbp-60h]
  __int64 v69; // [rsp+20h] [rbp-60h]
  unsigned __int8 v70; // [rsp+20h] [rbp-60h]
  __int64 v71; // [rsp+28h] [rbp-58h]
  __int64 v72; // [rsp+38h] [rbp-48h]
  __int64 *v73; // [rsp+40h] [rbp-40h] BYREF
  __int16 v74; // [rsp+48h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  if ( v5 != a2 + 72 )
  {
    v7 = 0;
    while ( 1 )
    {
      v8 = a1[3];
      if ( v5 )
      {
        v72 = v5 - 24;
        v9 = (unsigned int)(*(_DWORD *)(v5 + 20) + 1);
        v10 = *(_DWORD *)(v5 + 20) + 1;
      }
      else
      {
        v72 = 0;
        v9 = 0;
        v10 = 0;
      }
      if ( v10 < *(_DWORD *)(v8 + 32) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v9) )
        {
          v74 = 1;
          v11 = *(_QWORD *)(v72 + 56);
          v12 = v72 + 48;
          v73 = (__int64 *)v11;
          if ( v11 != v72 + 48 )
            break;
        }
      }
LABEL_17:
      v5 = *(_QWORD *)(v5 + 8);
      if ( a2 + 72 == v5 )
        return v7;
    }
    v71 = v5;
LABEL_11:
    v14 = *(__int64 **)(v11 + 8);
    v15 = v11 - 24;
    v74 = 0;
    v73 = v14;
    v16 = *(unsigned __int8 *)(v11 - 24);
    if ( (_BYTE)v16 == 62 )
    {
      v13 = sub_28B3870(a1, (unsigned __int8 *)(v11 - 24), (__int64)&v73);
      v11 = (unsigned __int64)v73;
      v7 |= v13;
      goto LABEL_10;
    }
    if ( (_BYTE)v16 == 85 )
    {
      v19 = *(_QWORD *)(v11 - 56);
      if ( !v19 )
      {
        v20 = 0;
        if ( *(char *)(v11 - 17) >= 0 )
          goto LABEL_51;
LABEL_27:
        v65 = v20;
        v68 = v15;
        v21 = sub_BD2BC0(v15);
        v15 = v68;
        v20 = v65;
        v23 = v21 + v22;
        if ( *(char *)(v11 - 17) >= 0 )
        {
          if ( (unsigned int)(v23 >> 4) )
LABEL_117:
            BUG();
        }
        else
        {
          v24 = sub_BD2BC0(v68);
          v15 = v68;
          v20 = v65;
          if ( (unsigned int)((v23 - v24) >> 4) )
          {
            v69 = v65;
            if ( *(char *)(v11 - 17) >= 0 )
              goto LABEL_117;
            v66 = v15;
            v25 = *(_DWORD *)(sub_BD2BC0(v15) + 8);
            if ( *(char *)(v11 - 17) >= 0 )
              BUG();
            v26 = sub_BD2BC0(v66);
            v20 = v69;
            v15 = v66;
            v28 = 32LL * (unsigned int)(*(_DWORD *)(v26 + v27 - 4) - v25);
            goto LABEL_32;
          }
        }
LABEL_51:
        v28 = 0;
LABEL_32:
        v29 = 0;
        v64 = (32LL * (*(_DWORD *)(v11 - 20) & 0x7FFFFFF) - 32 - v20 - v28) >> 5;
        if ( !v64 )
          goto LABEL_75;
        v67 = a1;
        v30 = v15;
        v63 = v12;
        v70 = v7;
        while ( 1 )
        {
          if ( (unsigned __int8)sub_B49B80(v30, v29, 81) )
          {
            v70 |= sub_28B06C0(v67, v30, v29, a3, a4, a5);
            goto LABEL_35;
          }
          v31 = *(unsigned __int8 *)(v11 - 24);
          if ( v31 == 40 )
          {
            v32 = 32LL * (unsigned int)sub_B491D0(v30);
          }
          else
          {
            v32 = 0;
            if ( v31 != 85 )
            {
              if ( v31 != 34 )
                goto LABEL_118;
              v32 = 64;
            }
          }
          if ( *(char *)(v11 - 17) >= 0 )
            goto LABEL_83;
          v35 = sub_BD2BC0(v30);
          v37 = v35 + v36;
          if ( *(char *)(v11 - 17) >= 0 )
            break;
          if ( !(unsigned int)((v37 - sub_BD2BC0(v30)) >> 4) )
            goto LABEL_83;
          if ( *(char *)(v11 - 17) >= 0 )
            goto LABEL_116;
          v38 = *(_DWORD *)(sub_BD2BC0(v30) + 8);
          if ( *(char *)(v11 - 17) >= 0 )
            BUG();
          v39 = sub_BD2BC0(v30);
          v41 = 32LL * (unsigned int)(*(_DWORD *)(v39 + v40 - 4) - v38);
LABEL_59:
          if ( v29 < (unsigned int)((32LL * (*(_DWORD *)(v11 - 20) & 0x7FFFFFF) - 32 - v32 - v41) >> 5)
            && (unsigned __int8)sub_B49B80(v30, v29, 81) )
          {
            goto LABEL_73;
          }
          v42 = *(unsigned __int8 *)(v11 - 24);
          if ( v42 == 40 )
          {
            v43 = 32LL * (unsigned int)sub_B491D0(v30);
          }
          else
          {
            v43 = 0;
            if ( v42 != 85 )
            {
              if ( v42 != 34 )
                goto LABEL_118;
              v43 = 64;
            }
          }
          if ( *(char *)(v11 - 17) < 0 )
          {
            v44 = sub_BD2BC0(v30);
            v46 = v44 + v45;
            if ( *(char *)(v11 - 17) >= 0 )
            {
              if ( (unsigned int)(v46 >> 4) )
LABEL_120:
                BUG();
            }
            else if ( (unsigned int)((v46 - sub_BD2BC0(v30)) >> 4) )
            {
              if ( *(char *)(v11 - 17) >= 0 )
                goto LABEL_120;
              v47 = *(_DWORD *)(sub_BD2BC0(v30) + 8);
              if ( *(char *)(v11 - 17) >= 0 )
                BUG();
              v48 = sub_BD2BC0(v30);
              v50 = 32LL * (unsigned int)(*(_DWORD *)(v48 + v49 - 4) - v47);
              goto LABEL_71;
            }
          }
          v50 = 0;
LABEL_71:
          if ( v29 >= (unsigned int)((32LL * (*(_DWORD *)(v11 - 20) & 0x7FFFFFF) - 32 - v43 - v50) >> 5) )
          {
            v52 = (_DWORD *)sub_B49810(v30, v29);
            if ( !*(_DWORD *)(*(_QWORD *)v52 + 8LL)
              && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v30
                                                  + 32
                                                  * (v29
                                                   - v52[2]
                                                   + (unsigned int)v52[2]
                                                   - (unsigned __int64)(*(_DWORD *)(v11 - 20) & 0x7FFFFFF)))
                                      + 8LL)
                          + 8LL) == 14 )
            {
              goto LABEL_73;
            }
LABEL_78:
            v53 = *(unsigned __int8 *)(v11 - 24);
            if ( v53 == 40 )
            {
              v54 = 32LL * (unsigned int)sub_B491D0(v30);
            }
            else
            {
              v54 = 0;
              if ( v53 != 85 )
              {
                if ( v53 != 34 )
                  goto LABEL_118;
                v54 = 64;
              }
            }
            if ( *(char *)(v11 - 17) >= 0 )
              goto LABEL_99;
            v55 = sub_BD2BC0(v30);
            v57 = v55 + v56;
            if ( *(char *)(v11 - 17) >= 0 )
            {
              if ( (unsigned int)(v57 >> 4) )
LABEL_123:
                BUG();
LABEL_99:
              v61 = 0;
              goto LABEL_95;
            }
            if ( !(unsigned int)((v57 - sub_BD2BC0(v30)) >> 4) )
              goto LABEL_99;
            if ( *(char *)(v11 - 17) >= 0 )
              goto LABEL_123;
            v58 = *(_DWORD *)(sub_BD2BC0(v30) + 8);
            if ( *(char *)(v11 - 17) >= 0 )
              BUG();
            v59 = sub_BD2BC0(v30);
            v61 = 32LL * (unsigned int)(*(_DWORD *)(v59 + v60 - 4) - v58);
LABEL_95:
            if ( v29 >= (unsigned int)((32LL * (*(_DWORD *)(v11 - 20) & 0x7FFFFFF) - 32 - v54 - v61) >> 5) )
            {
              sub_B49810(v30, v29);
            }
            else if ( (unsigned __int8)sub_B49B80(v30, v29, 50) )
            {
              goto LABEL_73;
            }
LABEL_35:
            if ( ++v29 == v64 )
              goto LABEL_74;
          }
          else
          {
            if ( !(unsigned __int8)sub_B49B80(v30, v29, 51) )
              goto LABEL_78;
LABEL_73:
            v51 = v29++;
            v70 |= sub_28B0E80(v67, v30, v51, a3, a4, a5);
            if ( v29 == v64 )
            {
LABEL_74:
              v12 = v63;
              v7 = v70;
              a1 = v67;
LABEL_75:
              v11 = (unsigned __int64)v73;
LABEL_10:
              if ( v12 == v11 )
              {
                v5 = v71;
                goto LABEL_17;
              }
              goto LABEL_11;
            }
          }
        }
        if ( (unsigned int)(v37 >> 4) )
LABEL_116:
          BUG();
LABEL_83:
        v41 = 0;
        goto LABEL_59;
      }
      if ( !*(_BYTE *)v19
        && *(_QWORD *)(v19 + 24) == *(_QWORD *)(v11 + 56)
        && (*(_BYTE *)(v19 + 33) & 0x20) != 0
        && ((*(_DWORD *)(v19 + 36) - 243) & 0xFFFFFFFD) == 0 )
      {
        v33 = sub_28ADB10((__int64)a1, v11 - 24, (__int64)&v73);
LABEL_45:
        v11 = (unsigned __int64)v73;
        if ( (_BYTE)v33 )
        {
          v7 = v33;
          if ( *(__int64 **)(v72 + 56) != v73 )
          {
            v34 = *v73;
            v74 = 0;
            v73 = (__int64 *)(v34 & 0xFFFFFFFFFFFFFFF8LL);
            v11 = v34 & 0xFFFFFFFFFFFFFFF8LL;
          }
        }
        goto LABEL_10;
      }
      if ( !*(_BYTE *)v19
        && *(_QWORD *)(v19 + 24) == *(_QWORD *)(v11 + 56)
        && (*(_BYTE *)(v19 + 33) & 0x20) != 0
        && ((*(_DWORD *)(v19 + 36) - 238) & 0xFFFFFFFD) == 0 )
      {
        v33 = sub_28B0630((__int64)a1, v11 - 24, &v73);
        goto LABEL_45;
      }
      v20 = 0;
      if ( !*(_BYTE *)v19
        && *(_QWORD *)(v19 + 24) == *(_QWORD *)(v11 + 56)
        && (*(_BYTE *)(v19 + 33) & 0x20) != 0
        && *(_DWORD *)(v19 + 36) == 241 )
      {
        v33 = sub_28ACCF0((__int64)a1, v11 - 24, &v73);
        goto LABEL_45;
      }
    }
    else
    {
      if ( (unsigned __int8)(v16 - 34) > 0x33u || (v17 = 0x8000000000041LL, !_bittest64(&v17, (unsigned int)(v16 - 34))) )
      {
        v11 = (unsigned __int64)v14;
        goto LABEL_10;
      }
      if ( v16 == 40 )
      {
        v62 = sub_B491D0(v11 - 24);
        v15 = v11 - 24;
        v20 = 32LL * v62;
      }
      else
      {
        if ( v16 != 34 )
LABEL_118:
          BUG();
        v20 = 64;
      }
    }
    if ( *(char *)(v11 - 17) >= 0 )
      goto LABEL_51;
    goto LABEL_27;
  }
  return 0;
}
