// Function: sub_2950CC0
// Address: 0x2950cc0
//
__int64 __fastcall sub_2950CC0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  char v8; // al
  unsigned int v9; // r13d
  unsigned __int8 v11; // r10
  __int64 v12; // rsi
  unsigned int v13; // r13d
  __int64 v14; // rax
  __int64 v16; // r15
  bool v17; // al
  bool v18; // al
  unsigned int v19; // r15d
  unsigned int v20; // r11d
  char *v21; // rax
  char *v22; // rax
  char *v23; // rdx
  _QWORD *v24; // rdx
  _QWORD *v25; // rdx
  _QWORD *v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // edi
  __int64 v31; // rax
  bool v32; // al
  int v33; // eax
  unsigned __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 i; // rdx
  char *v37; // rax
  unsigned int v38; // edx
  bool v39; // zf
  char *v40; // rax
  unsigned __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 j; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rcx
  unsigned int v46; // eax
  char *v47; // rdx
  unsigned int v48; // [rsp+Ch] [rbp-74h]
  unsigned int v49; // [rsp+Ch] [rbp-74h]
  unsigned int v50; // [rsp+10h] [rbp-70h]
  unsigned int v51; // [rsp+10h] [rbp-70h]
  unsigned int v52; // [rsp+10h] [rbp-70h]
  unsigned int v53; // [rsp+10h] [rbp-70h]
  unsigned int v54; // [rsp+10h] [rbp-70h]
  unsigned int v55; // [rsp+10h] [rbp-70h]
  char *v56; // [rsp+10h] [rbp-70h]
  unsigned int v57; // [rsp+10h] [rbp-70h]
  unsigned __int8 v59; // [rsp+18h] [rbp-68h]
  unsigned __int8 v60; // [rsp+18h] [rbp-68h]
  unsigned int v61; // [rsp+18h] [rbp-68h]
  unsigned __int8 v62; // [rsp+18h] [rbp-68h]
  unsigned int v63; // [rsp+18h] [rbp-68h]
  unsigned int v64; // [rsp+18h] [rbp-68h]
  unsigned int v65; // [rsp+18h] [rbp-68h]
  char *v66; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v67; // [rsp+28h] [rbp-58h]
  char *v68; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v69; // [rsp+38h] [rbp-48h]
  char *v70; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v71; // [rsp+48h] [rbp-38h]

  v8 = *(_BYTE *)a3;
  v9 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8;
  if ( (unsigned __int8)(*(_BYTE *)a3 - 22) > 6u )
  {
    v67 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8;
    v11 = a4;
    if ( v9 > 0x40 )
    {
      v48 = a6;
      v50 = a5;
      sub_C43690((__int64)&v66, 0, 0);
      v8 = *(_BYTE *)a3;
      v11 = a4;
      a5 = v50;
      a6 = v48;
      if ( *(_BYTE *)a3 == 17 )
      {
        v12 = a3 + 24;
        if ( v67 > 0x40 )
        {
LABEL_5:
          sub_C43990((__int64)&v66, v12);
          v13 = v67;
          goto LABEL_6;
        }
LABEL_4:
        if ( *(_DWORD *)(a3 + 32) <= 0x40u )
        {
          v23 = *(char **)(a3 + 24);
          v67 = *(_DWORD *)(a3 + 32);
          v66 = v23;
          goto LABEL_45;
        }
        goto LABEL_5;
      }
    }
    else
    {
      v66 = 0;
      v12 = a3 + 24;
      if ( v8 == 17 )
        goto LABEL_4;
    }
    if ( (unsigned __int8)(v8 - 42) <= 0x11u )
    {
      if ( (v8 & 0xEF) != 0x2A && v8 != 44 )
        goto LABEL_39;
      v16 = *(_QWORD *)(a3 - 64);
      if ( v8 == 58 )
      {
        if ( (*(_BYTE *)(a3 + 1) & 2) != 0 )
          goto LABEL_26;
        goto LABEL_39;
      }
      if ( ((unsigned __int8)a5 & (v11 ^ 1)) != 0 && v8 == 44 )
        goto LABEL_39;
      if ( v8 == 42 )
      {
        if ( (_BYTE)a5 != 1 && (_BYTE)a6 )
        {
          if ( *(_BYTE *)v16 == 17 )
          {
            v27 = *(_DWORD *)(v16 + 32);
            v28 = v27 > 0x40 ? *(_QWORD *)(*(_QWORD *)(v16 + 24) + 8LL * ((v27 - 1) >> 6)) : *(_QWORD *)(v16 + 24);
            if ( (v28 & (1LL << ((unsigned __int8)v27 - 1))) == 0 )
              goto LABEL_26;
          }
          v29 = *(_QWORD *)(a3 - 32);
          if ( *(_BYTE *)v29 == 17 )
          {
            v30 = *(_DWORD *)(v29 + 32);
            v31 = v30 > 0x40 ? *(_QWORD *)(*(_QWORD *)(v29 + 24) + 8LL * ((v30 - 1) >> 6)) : *(_QWORD *)(v29 + 24);
            if ( (v31 & (1LL << ((unsigned __int8)v30 - 1))) == 0 )
              goto LABEL_26;
          }
          if ( !v11 || (v54 = a5, v62 = v11, v32 = sub_B44900(a3), v11 = v62, a5 = v54, v32) )
          {
LABEL_26:
            v53 = (unsigned __int8)a5;
            v61 = v11;
            v49 = *(_DWORD *)(a2 + 8);
            sub_2950CC0(&v68, a2, v16, v11, (unsigned __int8)a5, 0);
            v19 = v69;
            v20 = v61;
            a5 = v53;
            if ( v69 > 0x40 )
            {
              v55 = v61;
              v63 = a5;
              v33 = sub_C444A0((__int64)&v68);
              a5 = v63;
              v20 = v55;
              if ( v19 - v33 > 0x40 )
                goto LABEL_29;
              v21 = *(char **)v68;
            }
            else
            {
              v21 = v68;
            }
            if ( v21 )
              goto LABEL_29;
            v34 = *(unsigned int *)(a2 + 8);
            if ( v49 != v34 )
            {
              if ( v49 >= v34 )
              {
                if ( v49 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                {
                  v57 = v20;
                  v65 = a5;
                  sub_C8D5F0(a2, (const void *)(a2 + 16), v49, 8u, a5, a6);
                  v34 = *(unsigned int *)(a2 + 8);
                  v20 = v57;
                  a5 = v65;
                }
                v35 = (_QWORD *)(*(_QWORD *)a2 + 8 * v34);
                for ( i = *(_QWORD *)a2 + 8LL * v49; (_QWORD *)i != v35; ++v35 )
                {
                  if ( v35 )
                    *v35 = 0;
                }
              }
              *(_DWORD *)(a2 + 8) = v49;
            }
            sub_2950CC0(&v70, a2, *(_QWORD *)(a3 - 32), v20, a5, 0);
            if ( v69 > 0x40 && v68 )
              j_j___libc_free_0_0((unsigned __int64)v68);
            v37 = v70;
            v38 = v71;
            v39 = *(_BYTE *)a3 == 44;
            v68 = v70;
            v69 = v71;
            if ( !v39 )
            {
LABEL_96:
              if ( v38 <= 0x40 )
              {
                v40 = v68;
LABEL_98:
                if ( !v40 )
                {
                  v41 = *(unsigned int *)(a2 + 8);
                  if ( v49 != v41 )
                  {
                    if ( v49 >= v41 )
                    {
                      if ( v49 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                      {
                        sub_C8D5F0(a2, (const void *)(a2 + 16), v49, 8u, a5, a6);
                        v41 = *(unsigned int *)(a2 + 8);
                      }
                      v42 = (_QWORD *)(*(_QWORD *)a2 + 8 * v41);
                      for ( j = *(_QWORD *)a2 + 8LL * v49; (_QWORD *)j != v42; ++v42 )
                      {
                        if ( v42 )
                          *v42 = 0;
                      }
                    }
                    *(_DWORD *)(a2 + 8) = v49;
                  }
                }
                goto LABEL_29;
              }
              if ( v38 - (unsigned int)sub_C444A0((__int64)&v68) <= 0x40 )
              {
                v40 = *(char **)v68;
                goto LABEL_98;
              }
LABEL_29:
              if ( v67 > 0x40 && v66 )
                j_j___libc_free_0_0((unsigned __int64)v66);
              v13 = v69;
              v66 = v68;
              v67 = v69;
LABEL_6:
              if ( v13 > 0x40 )
              {
                if ( v13 - (unsigned int)sub_C444A0((__int64)&v66) > 0x40 )
                  goto LABEL_8;
                v22 = *(char **)v66;
LABEL_34:
                if ( !v22 )
                {
LABEL_11:
                  *(_DWORD *)(a1 + 8) = v67;
                  *(_QWORD *)a1 = v66;
                  return a1;
                }
LABEL_8:
                v14 = *(unsigned int *)(a2 + 8);
                if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                {
                  sub_C8D5F0(a2, (const void *)(a2 + 16), v14 + 1, 8u, a5, a6);
                  v14 = *(unsigned int *)(a2 + 8);
                }
                *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = a3;
                ++*(_DWORD *)(a2 + 8);
                goto LABEL_11;
              }
LABEL_45:
              v22 = v66;
              goto LABEL_34;
            }
            if ( v71 > 0x40 )
            {
              sub_C43780((__int64)&v70, (const void **)&v68);
              v38 = v71;
              if ( v71 > 0x40 )
              {
                sub_C43D10((__int64)&v70);
LABEL_116:
                sub_C46250((__int64)&v70);
                v46 = v71;
                v71 = 0;
                v47 = v70;
                if ( v69 > 0x40 && v68 )
                {
                  v56 = v70;
                  v64 = v46;
                  j_j___libc_free_0_0((unsigned __int64)v68);
                  v47 = v56;
                  v46 = v64;
                }
                v68 = v47;
                v69 = v46;
                if ( v71 > 0x40 && v70 )
                  j_j___libc_free_0_0((unsigned __int64)v70);
                v38 = v69;
                goto LABEL_96;
              }
              v37 = v70;
            }
            v44 = ~(unsigned __int64)v37;
            v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v38;
            if ( !v38 )
              v45 = 0;
            v70 = (char *)(v45 & v44);
            goto LABEL_116;
          }
LABEL_39:
          v13 = v67;
          goto LABEL_6;
        }
      }
      else if ( v8 != 44 )
      {
        goto LABEL_26;
      }
      if ( !v11 || (v51 = a5, v59 = v11, v17 = sub_B44900(a3), v11 = v59, a5 = v51, v17) )
      {
        if ( !(_BYTE)a5 )
          goto LABEL_26;
        v52 = a5;
        v60 = v11;
        v18 = sub_B448F0(a3);
        v11 = v60;
        a5 = v52;
        if ( v18 )
          goto LABEL_26;
      }
      goto LABEL_39;
    }
    switch ( v8 )
    {
      case 'C':
        if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
          v25 = *(_QWORD **)(a3 - 8);
        else
          v25 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
        sub_2950CC0(&v68, a2, *v25, v11, (unsigned __int8)a5, (unsigned __int8)a6);
        sub_C44740((__int64)&v70, &v68, v9);
        if ( v67 > 0x40 )
          goto LABEL_50;
        break;
      case 'E':
        if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
          v24 = *(_QWORD **)(a3 - 8);
        else
          v24 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
        sub_2950CC0(&v68, a2, *v24, 1, (unsigned __int8)a5, (unsigned __int8)a6);
        sub_C44830((__int64)&v70, &v68, v9);
        if ( v67 > 0x40 )
        {
LABEL_50:
          if ( v66 )
            j_j___libc_free_0_0((unsigned __int64)v66);
        }
        break;
      case 'D':
        if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
          v26 = *(_QWORD **)(a3 - 8);
        else
          v26 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
        sub_2950CC0(&v68, a2, *v26, 0, 1, 0);
        sub_C449B0((__int64)&v70, (const void **)&v68, v9);
        if ( v67 <= 0x40 )
          break;
        goto LABEL_50;
      default:
        goto LABEL_39;
    }
    v13 = v71;
    v66 = v70;
    v67 = v71;
    if ( v69 <= 0x40 || !v68 )
      goto LABEL_6;
    j_j___libc_free_0_0((unsigned __int64)v68);
    goto LABEL_39;
  }
  *(_DWORD *)(a1 + 8) = v9;
  if ( v9 > 0x40 )
    sub_C43690(a1, 0, 0);
  else
    *(_QWORD *)a1 = 0;
  return a1;
}
