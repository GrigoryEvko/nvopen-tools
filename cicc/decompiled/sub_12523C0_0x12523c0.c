// Function: sub_12523C0
// Address: 0x12523c0
//
__int64 __fastcall sub_12523C0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 result; // rax
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // ecx
  unsigned int v9; // edi
  __int64 *v10; // rdx
  __int64 v11; // r10
  int v12; // r9d
  __int64 *v13; // r10
  unsigned int v14; // r11d
  __int64 *v15; // rdx
  __int64 v16; // rdi
  size_t v17; // r10
  const char *v18; // r12
  __int64 v19; // r15
  _BYTE *v20; // rax
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rdx
  const char *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned __int64 v30; // rax
  unsigned int v31; // eax
  int v32; // eax
  unsigned int v33; // eax
  size_t v34; // r10
  void *v35; // rax
  __int64 v36; // rdi
  __int64 *v37; // r13
  __int64 *v38; // r12
  __int64 v39; // rdx
  _QWORD *v40; // rdx
  __int64 v41; // rbx
  int v42; // eax
  __int64 v43; // rcx
  int v44; // esi
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdi
  _QWORD *v50; // rax
  unsigned __int64 v51; // rdx
  const char *v52; // rax
  int v53; // ecx
  int v54; // ecx
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rsi
  int v58; // r8d
  __int64 *v59; // r12
  __int64 v60; // rax
  int v61; // eax
  int v62; // r8d
  __int64 v63; // rdi
  _QWORD *v64; // rax
  void *v65; // rax
  int v66; // edx
  int v67; // r9d
  int v68; // edx
  int v69; // edx
  __int64 v70; // rsi
  __int64 *v71; // r11
  __int64 v72; // r12
  int v73; // r8d
  __int64 v74; // rcx
  __int64 v75; // [rsp+0h] [rbp-C0h]
  size_t v77; // [rsp+18h] [rbp-A8h]
  size_t v78; // [rsp+18h] [rbp-A8h]
  size_t v79; // [rsp+18h] [rbp-A8h]
  const char *v80; // [rsp+20h] [rbp-A0h]
  size_t v81; // [rsp+28h] [rbp-98h]
  unsigned __int64 v82; // [rsp+28h] [rbp-98h]
  _QWORD v83[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v84; // [rsp+50h] [rbp-70h]
  const char *v85[2]; // [rsp+60h] [rbp-60h] BYREF
  const char *v86; // [rsp+70h] [rbp-50h]
  unsigned __int64 v87; // [rsp+78h] [rbp-48h]
  __int16 v88; // [rsp+80h] [rbp-40h]

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 208);
  result = v3 + 40LL * *(unsigned int *)(a1 + 216);
  v75 = result;
  if ( v3 != result )
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v3 + 24);
      v18 = *(const char **)(v3 + 16);
      v19 = *(_QWORD *)(v3 + 8);
      if ( !v17 )
        break;
      v81 = *(_QWORD *)(v3 + 24);
      v20 = memchr(*(const void **)(v3 + 16), 64, v81);
      v17 = v81;
      v21 = v20 - v18;
      if ( !v20 )
        v21 = -1;
      if ( v21 > v81 )
        break;
      v80 = &v18[v21];
      v82 = v81 - v21;
      if ( v17 - v21 != -1 && v82 <= 2 )
      {
        v22 = v82;
        v23 = &v18[v21];
        goto LABEL_50;
      }
      if ( *(_WORD *)v80 == 16448 && v80[2] == 64 )
      {
        if ( *(_QWORD *)v19
          || (*(_BYTE *)(v19 + 9) & 0x70) == 0x20
          && *(char *)(v19 + 8) >= 0
          && (*(_BYTE *)(v19 + 8) |= 8u,
              v79 = v17,
              v65 = sub_E807D0(*(_QWORD *)(v19 + 24)),
              v17 = v79,
              (*(_QWORD *)v19 = v65) != 0) )
        {
          v60 = 1;
        }
        else
        {
          v60 = 2;
        }
        v22 = v82 - v60;
        v23 = &v80[v60];
      }
      else
      {
        v22 = v82;
        v23 = &v18[v21];
      }
LABEL_17:
      v24 = *a2;
      v77 = v17;
      v88 = 1285;
      v87 = v22;
      v85[1] = (const char *)v21;
      v85[0] = v18;
      v86 = v23;
      v25 = sub_E6C460(v24, v85);
      sub_E5CB20((__int64)a2, v25, v26, v27, v28, v29);
      v30 = sub_E808D0(v19, 0, (_QWORD *)*a2, 0);
      sub_EA12A0(v25, v30);
      v31 = sub_EA1780(v19);
      sub_EA1710(v25, v31);
      v32 = sub_EA1680(v19);
      sub_EA1660(v25, v32);
      v33 = sub_EA16B0(v19);
      result = sub_EA1690(v25, v33);
      if ( *(_QWORD *)v19 )
        goto LABEL_3;
      v34 = v77;
      if ( (*(_BYTE *)(v19 + 9) & 0x70) == 0x20 )
      {
        if ( *(char *)(v19 + 8) < 0 )
          goto LABEL_22;
        *(_BYTE *)(v19 + 8) |= 8u;
        result = (__int64)sub_E807D0(*(_QWORD *)(v19 + 24));
        v34 = v77;
        *(_QWORD *)v19 = result;
        if ( result )
        {
LABEL_3:
          if ( *(_BYTE *)(v3 + 32) )
            goto LABEL_9;
          goto LABEL_4;
        }
        if ( (*(_BYTE *)(v19 + 9) & 0x70) == 0x20 )
        {
LABEL_22:
          if ( *(char *)(v19 + 8) >= 0 )
          {
            *(_BYTE *)(v19 + 8) |= 8u;
            v78 = v34;
            v35 = sub_E807D0(*(_QWORD *)(v19 + 24));
            v34 = v78;
            *(_QWORD *)v19 = v35;
            if ( v35 )
              goto LABEL_4;
          }
        }
      }
      if ( v82 <= 1 || *(_WORD *)v80 != 16448 || v82 != 2 && v80[2] == 64 )
      {
LABEL_4:
        v6 = *(_DWORD *)(a1 + 192);
        result = *(_QWORD *)(a1 + 176);
        v7 = a1 + 168;
        if ( !v6 )
        {
          ++*(_QWORD *)(a1 + 168);
          goto LABEL_57;
        }
        v8 = v6 - 1;
        v9 = (v6 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v10 = (__int64 *)(result + 16LL * v9);
        v11 = *v10;
        if ( v19 == *v10 )
        {
LABEL_6:
          if ( v10 != (__int64 *)(result + 16LL * v6) && v25 != v10[1] )
          {
            v49 = *a2;
            if ( (*(_BYTE *)(v19 + 8) & 1) != 0 )
            {
              v50 = *(_QWORD **)(v19 - 8);
              v51 = *v50;
              v52 = (const char *)(v50 + 3);
            }
            else
            {
              v51 = 0;
              v52 = 0;
            }
            v86 = v52;
            v87 = v51;
            v85[0] = "multiple versions for ";
            v88 = 1283;
            result = sub_E66880(v49, *(_QWORD **)v3, (__int64)v85);
            goto LABEL_9;
          }
        }
        else
        {
          v66 = 1;
          while ( v11 != -4096 )
          {
            v67 = v66 + 1;
            v9 = v8 & (v66 + v9);
            v10 = (__int64 *)(result + 16LL * v9);
            v11 = *v10;
            if ( v19 == *v10 )
              goto LABEL_6;
            v66 = v67;
          }
        }
        v12 = 1;
        v13 = 0;
        v14 = v8 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v15 = (__int64 *)(result + 16LL * v14);
        v16 = *v15;
        if ( v19 != *v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 != -8192 || v13 )
              v15 = v13;
            v14 = v8 & (v12 + v14);
            v16 = *(_QWORD *)(result + 16LL * v14);
            if ( v19 == v16 )
              goto LABEL_9;
            ++v12;
            v13 = v15;
            v15 = (__int64 *)(result + 16LL * v14);
          }
          if ( !v13 )
            v13 = v15;
          ++*(_QWORD *)(a1 + 168);
          result = (unsigned int)(*(_DWORD *)(a1 + 184) + 1);
          if ( 4 * (int)result < 3 * v6 )
          {
            if ( v6 - *(_DWORD *)(a1 + 188) - (unsigned int)result <= v6 >> 3 )
            {
              sub_12521E0(v7, v6);
              v68 = *(_DWORD *)(a1 + 192);
              if ( !v68 )
              {
LABEL_114:
                ++*(_DWORD *)(a1 + 184);
                BUG();
              }
              v69 = v68 - 1;
              v70 = *(_QWORD *)(a1 + 176);
              v71 = 0;
              LODWORD(v72) = v69 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
              v73 = 1;
              result = (unsigned int)(*(_DWORD *)(a1 + 184) + 1);
              v13 = (__int64 *)(v70 + 16LL * (unsigned int)v72);
              v74 = *v13;
              if ( v19 != *v13 )
              {
                while ( v74 != -4096 )
                {
                  if ( !v71 && v74 == -8192 )
                    v71 = v13;
                  v72 = v69 & (unsigned int)(v72 + v73);
                  v13 = (__int64 *)(v70 + 16 * v72);
                  v74 = *v13;
                  if ( v19 == *v13 )
                    goto LABEL_89;
                  ++v73;
                }
                if ( v71 )
                  v13 = v71;
              }
            }
            goto LABEL_89;
          }
LABEL_57:
          sub_12521E0(v7, 2 * v6);
          v53 = *(_DWORD *)(a1 + 192);
          if ( !v53 )
            goto LABEL_114;
          v54 = v53 - 1;
          v55 = *(_QWORD *)(a1 + 176);
          LODWORD(v56) = v54 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          result = (unsigned int)(*(_DWORD *)(a1 + 184) + 1);
          v13 = (__int64 *)(v55 + 16LL * (unsigned int)v56);
          v57 = *v13;
          if ( v19 != *v13 )
          {
            v58 = 1;
            v59 = 0;
            while ( v57 != -4096 )
            {
              if ( !v59 && v57 == -8192 )
                v59 = v13;
              v56 = v54 & (unsigned int)(v56 + v58);
              v13 = (__int64 *)(v55 + 16 * v56);
              v57 = *v13;
              if ( v19 == *v13 )
                goto LABEL_89;
              ++v58;
            }
            if ( v59 )
              v13 = v59;
          }
LABEL_89:
          *(_DWORD *)(a1 + 184) = result;
          if ( *v13 != -4096 )
            --*(_DWORD *)(a1 + 188);
          *v13 = v19;
          v13[1] = v25;
        }
LABEL_9:
        v3 += 40;
        if ( v3 == v75 )
          goto LABEL_29;
      }
      else
      {
        v83[2] = v18;
        v36 = *a2;
        v3 += 40;
        v83[0] = "default version symbol ";
        v85[0] = (const char *)v83;
        v84 = 1283;
        v83[3] = v34;
        v86 = " must be defined";
        v88 = 770;
        result = sub_E66880(v36, *(_QWORD **)(v3 - 40), (__int64)v85);
        if ( v3 == v75 )
        {
LABEL_29:
          v2 = a1;
          goto LABEL_30;
        }
      }
    }
    v23 = &v18[v17];
    v21 = v17;
    v22 = 0;
LABEL_50:
    v80 = v23;
    v82 = v22;
    goto LABEL_17;
  }
LABEL_30:
  v37 = *(__int64 **)(v2 + 64);
  v38 = *(__int64 **)(v2 + 56);
  if ( v37 != v38 )
  {
    while ( 1 )
    {
      v42 = *(_DWORD *)(v2 + 192);
      v41 = *v38;
      v43 = *(_QWORD *)(v2 + 176);
      if ( v42 )
      {
        v44 = v42 - 1;
        v45 = (v42 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v46 = (__int64 *)(v43 + 16LL * v45);
        v47 = *v46;
        if ( v41 == *v46 )
        {
LABEL_41:
          v48 = v46[1];
          if ( v48 )
          {
            *v38 = v48;
            v41 = v48;
          }
        }
        else
        {
          v61 = 1;
          while ( v47 != -4096 )
          {
            v62 = v61 + 1;
            v45 = v44 & (v61 + v45);
            v46 = (__int64 *)(v43 + 16LL * v45);
            v47 = *v46;
            if ( v41 == *v46 )
              goto LABEL_41;
            v61 = v62;
          }
        }
      }
      result = *(_QWORD *)v41;
      if ( *(_QWORD *)v41 )
        break;
      result = *(_BYTE *)(v41 + 9) & 0x70;
      if ( (_BYTE)result == 32 && *(char *)(v41 + 8) >= 0 )
      {
        *(_BYTE *)(v41 + 8) |= 8u;
        result = (__int64)sub_E807D0(*(_QWORD *)(v41 + 24));
        *(_QWORD *)v41 = result;
        v41 = *v38;
        if ( result )
          break;
        ++v38;
        *(_BYTE *)(v41 + 9) |= 8u;
        if ( v37 == v38 )
          return result;
      }
      else
      {
LABEL_38:
        ++v38;
        *(_BYTE *)(v41 + 9) |= 8u;
        if ( v37 == v38 )
          return result;
      }
    }
    if ( off_4C5D170 != (_UNKNOWN *)result )
    {
      result = *(unsigned __int8 *)(v41 + 8);
      if ( (result & 1) != 0 )
      {
        v39 = *(_QWORD *)(v41 - 8);
        if ( *(_QWORD *)v39 > 1u && *(_WORD *)(v39 + 24) == 19502 )
        {
          v40 = *(_QWORD **)v41;
          if ( !*(_QWORD *)v41 )
          {
            if ( (*(_BYTE *)(v41 + 9) & 0x70) != 0x20 || (result & 0x80u) != 0LL )
              BUG();
            v63 = *(_QWORD *)(v41 + 24);
            *(_BYTE *)(v41 + 8) = result | 8;
            v64 = sub_E807D0(v63);
            *(_QWORD *)v41 = v64;
            v40 = v64;
          }
          result = v40[1];
          v41 = *(_QWORD *)(result + 16);
          *v38 = v41;
        }
      }
    }
    goto LABEL_38;
  }
  return result;
}
