// Function: sub_CBC140
// Address: 0xcbc140
//
__int64 __fastcall sub_CBC140(
        __int64 *a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7)
{
  __int64 v10; // rbx
  __int64 v11; // r9
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  int v14; // eax
  int v15; // eax
  __int64 result; // rax
  __int64 v17; // rax
  unsigned __int8 *v18; // rax
  unsigned __int64 v19; // rcx
  int v20; // eax
  int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // r11
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  unsigned __int8 **v26; // rax
  int v27; // eax
  __int64 v28; // r8
  __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 *v32; // rax
  int v33; // ecx
  __int64 v34; // rdx
  __int64 v35; // rax
  size_t v36; // rdx
  int v37; // esi
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 *v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+18h] [rbp-58h]
  unsigned __int8 v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+18h] [rbp-58h]
  unsigned __int8 v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+20h] [rbp-50h]
  unsigned __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  unsigned __int8 v58; // [rsp+28h] [rbp-48h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  unsigned __int8 v60; // [rsp+28h] [rbp-48h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  size_t v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 v64; // [rsp+30h] [rbp-40h]
  __int64 v65; // [rsp+30h] [rbp-40h]
  __int64 v66; // [rsp+30h] [rbp-40h]
  __int64 v67; // [rsp+30h] [rbp-40h]

  v10 = a5;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
LABEL_2:
          if ( a4 >= v10 )
          {
LABEL_108:
            result = (__int64)a2;
            if ( a2 == a3 )
              return result;
            return 0;
          }
          v64 = *a1;
          v11 = *(_QWORD *)(*a1 + 8);
          while ( 1 )
          {
            v12 = *(_QWORD *)(v11 + 8 * a4);
            v13 = (unsigned int)v12 & 0xF8000000;
            if ( v13 == 805306368 )
            {
              v22 = *(_QWORD *)(v64 + 24) + 32 * (v12 & 0x7FFFFFF);
              if ( a3 == a2 || (*(_BYTE *)(v22 + 8) & *(_BYTE *)(*(_QWORD *)v22 + *a2)) == 0 )
                return 0;
              ++a2;
              goto LABEL_9;
            }
            if ( v13 <= 0x30000000 )
              break;
            if ( v13 == 2415919104 )
              goto LABEL_9;
            if ( v13 <= 0x90000000 )
            {
              if ( v13 != 1610612736 )
              {
                if ( v13 != 0x80000000 )
                  goto LABEL_70;
                v17 = *(_QWORD *)(v11 + 8 * a4++ + 8);
                do
                {
                  a4 += v17 & 0x7FFFFFF;
                  v17 = *(_QWORD *)(v11 + 8 * a4);
                }
                while ( (v17 & 0xF8000000) != 0x90000000LL );
              }
            }
            else
            {
              if ( v13 == 2550136832 )
              {
                v18 = (unsigned __int8 *)a1[4];
                v19 = a1[5];
                if ( v18 != a2 )
                {
                  if ( v19 > (unsigned __int64)a2 && *(a2 - 1) == 10 )
                    goto LABEL_62;
                  goto LABEL_40;
                }
                if ( (a1[1] & 1) != 0 )
                {
                  if ( v19 <= (unsigned __int64)a2 || *(a2 - 1) != 10 )
                    return 0;
LABEL_62:
                  if ( (*(_BYTE *)(v64 + 40) & 8) == 0 )
                  {
LABEL_40:
                    v47 = a4;
                    v51 = v11;
                    v55 = a1[5];
                    if ( v18 >= a2 )
                      return 0;
                    v60 = *(a2 - 1);
                    v20 = isalnum(v60);
                    if ( v60 == 95 )
                      return 0;
                    v19 = v55;
                    v11 = v51;
                    a4 = v47;
                    if ( v20 )
                      return 0;
                    goto LABEL_43;
                  }
                }
                else
                {
LABEL_43:
                  if ( v19 <= (unsigned __int64)a2 )
                    return 0;
                }
                v56 = a4;
                v61 = v11;
                v52 = *a2;
                v21 = isalnum(*a2);
                v11 = v61;
                a4 = v56;
                if ( !v21 && v52 != 95 )
                  return 0;
                goto LABEL_9;
              }
              if ( v13 != 2684354560 )
                goto LABEL_70;
              if ( (unsigned __int8 *)a1[5] == a2 )
              {
                if ( (a1[1] & 2) != 0 )
                  return 0;
              }
              else
              {
                if ( a1[5] <= (unsigned __int64)a2 )
                  return 0;
                if ( *a2 != 10 || (*(_BYTE *)(v64 + 40) & 8) == 0 )
                {
                  v58 = *a2;
                  v49 = a4;
                  v53 = v11;
                  v14 = isalnum(*a2);
                  if ( v58 == 95 )
                    return 0;
                  v11 = v53;
                  a4 = v49;
                  if ( v14 )
                    return 0;
                }
              }
              v54 = a4;
              v59 = v11;
              if ( a1[4] >= (unsigned __int64)a2 )
                return 0;
              v50 = *(a2 - 1);
              v15 = isalnum(v50);
              v11 = v59;
              a4 = v54;
              if ( v50 != 95 && !v15 )
                return 0;
            }
LABEL_9:
            if ( v10 <= ++a4 )
              goto LABEL_108;
          }
          if ( v13 == 0x20000000 )
          {
            if ( (unsigned __int8 *)a1[5] != a2 )
            {
              if ( a1[5] <= (unsigned __int64)a2 || *a2 != 10 )
                return 0;
LABEL_36:
              if ( (*(_BYTE *)(v64 + 40) & 8) == 0 )
                return 0;
              goto LABEL_9;
            }
            if ( (a1[1] & 2) != 0 )
              return 0;
            goto LABEL_9;
          }
          if ( v13 > 0x20000000 )
          {
            if ( v13 != 671088640 )
              goto LABEL_70;
            if ( a3 == a2 )
              return 0;
            ++a2;
            goto LABEL_9;
          }
          if ( v13 == 0x10000000 )
          {
            if ( a3 == a2 || *a2 != (_BYTE)v12 )
              return 0;
            ++a2;
            goto LABEL_9;
          }
          if ( v13 == 402653184 )
          {
            if ( (unsigned __int8 *)a1[4] == a2 && (a1[1] & 1) == 0 )
              goto LABEL_9;
            if ( a1[5] <= (unsigned __int64)a2 || *(a2 - 1) != 10 )
              return 0;
            goto LABEL_36;
          }
LABEL_70:
          v23 = a4 + 1;
          v24 = a4 + 1;
          v65 = *(_QWORD *)(v11 + 8 * a4);
          v25 = (unsigned int)v65 & 0xF8000000;
          if ( v25 != 1476395008 )
            break;
          v63 = a4 + 1;
          result = sub_CBC140((_DWORD)a1, (_DWORD)a2, (_DWORD)a3, (int)a4 + 1, v10, a6, a7);
          if ( result )
            return result;
          a4 = v63 + (v65 & 0x7FFFFFF);
        }
        if ( v25 > 0x58000000 )
        {
          v28 = v10;
          if ( v25 == 1879048192 )
          {
            v42 = 16 * (v65 & 0x7FFFFFF);
            v43 = v42 + a1[2];
            v44 = *(_QWORD *)(v43 + 8);
            *(_QWORD *)(v43 + 8) = &a2[-a1[3]];
            result = sub_CBC140((_DWORD)a1, (_DWORD)a2, (_DWORD)a3, v23, v28, a6, a7);
            if ( !result )
              *(_QWORD *)(a1[2] + v42 + 8) = v44;
            return result;
          }
          v29 = v24 + (v65 & 0x7FFFFFF) - 2;
          if ( v25 == 2013265920 )
          {
            while ( 1 )
            {
              v67 = v28;
              result = sub_CBC140((_DWORD)a1, (_DWORD)a2, (_DWORD)a3, v24, v28, a6, a7);
              if ( result )
                break;
              v30 = *(_QWORD *)(*a1 + 8);
              if ( (*(_QWORD *)(v30 + 8 * v29) & 0xF8000000LL) == 0x90000000LL )
                return 0;
              LODWORD(v24) = v29 + 2;
              v28 = v67;
              v31 = v29 + (*(_QWORD *)(v30 + 8 * v29 + 8) & 0x7FFFFFFLL) + 1;
              v29 = v31 - ((*(_QWORD *)(v30 + 8 * v31) & 0xF8000000LL) == 2281701376LL);
            }
            return result;
          }
          if ( v25 == 1744830464 )
          {
            v40 = (__int64 *)(16 * (v65 & 0x7FFFFFF) + a1[2]);
            v41 = *v40;
            *v40 = (__int64)&a2[-a1[3]];
            result = sub_CBC140((_DWORD)a1, (_DWORD)a2, (_DWORD)a3, v23, v28, a6, a7);
            if ( !result )
              *(_QWORD *)(a1[2] + 16 * (v65 & 0x7FFFFFF)) = v41;
            return result;
          }
          return 0;
        }
        if ( v25 != 1207959552 )
          break;
        ++a4;
        *(_QWORD *)(a1[7] + 8 * ++a6) = a2;
      }
      if ( v25 == 1342177280 )
        break;
      if ( v25 != 939524096 )
        return 0;
      v32 = (__int64 *)(a1[2] + 16 * (v65 & 0x7FFFFFF));
      v33 = v65 & 0x7FFFFFF;
      v34 = v32[1];
      if ( v34 == -1 )
        return 0;
      v35 = *v32;
      v36 = v34 - v35;
      if ( v36 )
      {
        v45 = a4 + 1;
        v48 = a4;
        v57 = v11;
        if ( a2 > &a3[-v36] )
          return 0;
        v62 = v36;
        if ( memcmp(a2, (const void *)(a1[3] + v35), v36) )
          return 0;
        v37 = a7;
        v23 = v45;
        a4 = v48;
        v33 = v65 & 0x7FFFFFF;
        v11 = v57;
        v36 = v62;
      }
      else
      {
        v37 = a7 + 1;
        if ( a7 > 100 || a3 < a2 )
          return 0;
      }
      v38 = v33 | 0x40000000;
      if ( v65 == v38 )
      {
        a4 = v23;
      }
      else
      {
        while ( 1 )
        {
          v39 = a4;
          a4 = v23;
          if ( *(_QWORD *)(v11 + 8 * v23) == v38 )
            break;
          ++v23;
        }
        a4 = v39 + 2;
      }
      a7 = v37;
      a2 += v36;
    }
    v26 = (unsigned __int8 **)(a1[7] + 8 * a6);
    if ( *v26 != a2 )
      break;
    ++a4;
    --a6;
  }
  *v26 = a2;
  v27 = v65 & 0x7FFFFFF;
  v66 = a4 + 1;
  result = sub_CBC140((_DWORD)a1, (_DWORD)a2, (_DWORD)a3, (int)v23 - v27, v10, a6, a7);
  if ( !result )
  {
    --a6;
    a4 = v66;
    goto LABEL_2;
  }
  return result;
}
