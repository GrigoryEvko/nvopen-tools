// Function: sub_19B7FA0
// Address: 0x19b7fa0
//
__int64 __fastcall sub_19B7FA0(__int64 *a1, __int64 a2, int a3, __int64 a4, int a5, int a6)
{
  __int64 *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 result; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // r8d
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // r10
  int v21; // edi
  int v22; // esi
  int v23; // r11d
  __int64 v24; // r10
  __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 v27; // rbx
  const void *v28; // r14
  __int64 v29; // r8
  __int64 v30; // rdi
  __int64 v31; // r8
  char v32; // di
  unsigned int v33; // esi
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r13
  int v39; // r8d
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 *v44; // r15
  _QWORD *v45; // r14
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 v51; // r13
  _QWORD *v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 **v55; // r10
  __int64 v56; // rax
  __int64 *v57; // r13
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // r8
  _BYTE *v62; // rcx
  int v63; // esi
  _BYTE *v64; // r11
  __int64 *v65; // rax
  __int64 v66; // rsi
  int v67; // r13d
  _QWORD *v68; // rdx
  unsigned __int64 v69; // [rsp+0h] [rbp-90h]
  __int64 v70; // [rsp+8h] [rbp-88h]
  __int64 v71; // [rsp+10h] [rbp-80h]
  __int64 v73; // [rsp+20h] [rbp-70h]
  __int64 **v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+28h] [rbp-68h]
  _BYTE *v76; // [rsp+30h] [rbp-60h] BYREF
  __int64 v77; // [rsp+38h] [rbp-58h]
  _BYTE v78[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = a1;
  v7 = *a1;
  v8 = *(unsigned int *)(*a1 + 8);
  if ( (unsigned int)v8 >= *(_DWORD *)(*a1 + 12) )
  {
    sub_16CD150(v7, (const void *)(v7 + 16), 0, 8, a5, a6);
    v8 = *(unsigned int *)(v7 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v7 + 8 * v8) = a2;
  ++*(_DWORD *)(v7 + 8);
  while ( 2 )
  {
    v9 = *v6;
    LODWORD(result) = *(_DWORD *)(*v6 + 8);
LABEL_5:
    while ( 2 )
    {
      while ( 2 )
      {
        v11 = *(_QWORD *)(*(_QWORD *)v9 + 8LL * (unsigned int)result - 8);
        *(_DWORD *)(v9 + 8) = result - 1;
        v12 = v6[1];
        v13 = *(unsigned int *)(v12 + 24);
        if ( !(_DWORD)v13 )
          goto LABEL_13;
        v14 = *(_QWORD *)(v12 + 8);
        a6 = v13 - 1;
        v15 = (4 * a3) >> 2;
        v71 = (unsigned int)(v15 + 4 * (v15 + 8 * a3));
        v16 = (((v71 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32)) - 1 - (v71 << 32)) >> 22)
            ^ ((v71 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32)) - 1 - (v71 << 32));
        v17 = ((9 * (((v16 - 1 - (v16 << 13)) >> 8) ^ (v16 - 1 - (v16 << 13)))) >> 15)
            ^ (9 * (((v16 - 1 - (v16 << 13)) >> 8) ^ (v16 - 1 - (v16 << 13))));
        v18 = (v13 - 1) & (((v17 - 1 - (v17 << 27)) >> 31) ^ (v17 - 1 - ((_DWORD)v17 << 27)));
        v19 = (__int64 *)(v14 + 16LL * v18);
        v20 = *v19;
        v21 = (4 * *((_DWORD *)v19 + 2)) >> 2;
        if ( v21 != v15 || v11 != v20 )
        {
          v22 = 1;
          if ( v20 == -8 )
            goto LABEL_12;
          while ( 1 )
          {
            v23 = v22 + 1;
            v18 = a6 & (v22 + v18);
            v19 = (__int64 *)(v14 + 16LL * v18);
            v24 = *v19;
            v21 = (4 * *((_DWORD *)v19 + 2)) >> 2;
            if ( v21 == v15 && v11 == v24 )
              break;
            v22 = v23;
            if ( v24 == -8 )
            {
LABEL_12:
              if ( !v21 )
                goto LABEL_13;
            }
          }
        }
        if ( v19 == (__int64 *)(16 * v13 + v14) || *((char *)v19 + 11) < 0 )
          goto LABEL_13;
        *((_BYTE *)v19 + 11) |= 0x80u;
        if ( *(_BYTE *)(v11 + 16) != 77 || (v30 = *(_QWORD *)v6[2], **(_QWORD **)(v30 + 32) != *(_QWORD *)(v11 + 40)) )
        {
          if ( (*((_BYTE *)v19 + 11) & 0x40) == 0 )
          {
            v55 = (__int64 **)v6[5];
            v56 = 3LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
            {
              v57 = *(__int64 **)(v11 - 8);
              v58 = (__int64)&v57[v56];
            }
            else
            {
              v57 = (__int64 *)(v11 - v56 * 8);
              v58 = v11;
            }
            v59 = v58 - (_QWORD)v57;
            v77 = 0x400000000LL;
            v76 = v78;
            v60 = 0xAAAAAAAAAAAAAAABLL * (v59 >> 3);
            v61 = v60;
            if ( (unsigned __int64)v59 > 0x60 )
            {
              v74 = v55;
              v69 = 0xAAAAAAAAAAAAAAABLL * (v59 >> 3);
              v70 = v59;
              sub_16CD150((__int64)&v76, v78, v60, 8, v60, a6);
              v64 = v76;
              v63 = v77;
              LODWORD(v60) = v69;
              v55 = v74;
              v59 = v70;
              v61 = v69;
              v62 = &v76[8 * (unsigned int)v77];
            }
            else
            {
              v62 = v78;
              v63 = 0;
              v64 = v78;
            }
            if ( v59 > 0 )
            {
              v65 = v57;
              do
              {
                v66 = *v65;
                v62 += 8;
                v65 += 3;
                *((_QWORD *)v62 - 1) = v66;
                --v61;
              }
              while ( v61 );
              v64 = v76;
              v63 = v77;
            }
            LODWORD(v77) = v63 + v60;
            v67 = sub_14A5330(v55, v11, (__int64)v64, (unsigned int)(v63 + v60));
            if ( v76 != v78 )
              _libc_free((unsigned __int64)v76);
            *(_DWORD *)v6[4] += v67;
          }
          v42 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
          if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
          {
            v43 = *(_QWORD *)(v11 - 8);
            v11 = v43 + v42;
          }
          else
          {
            v43 = v11 - v42;
          }
          if ( v11 != v43 )
          {
            v75 = v11;
            v44 = v6;
            while ( 1 )
            {
              v50 = *(_QWORD *)v43;
              if ( *(_BYTE *)(*(_QWORD *)v43 + 16LL) > 0x17u )
                break;
LABEL_49:
              v43 += 24;
              if ( v75 == v43 )
              {
                v6 = v44;
                v9 = *v44;
                result = *(unsigned int *)(*v44 + 8);
                if ( !(_DWORD)result )
                  goto LABEL_14;
                goto LABEL_5;
              }
            }
            v51 = *(_QWORD *)v44[2];
            v52 = *(_QWORD **)(v51 + 72);
            v46 = *(_QWORD **)(v51 + 64);
            if ( v52 == v46 )
            {
              v45 = &v46[*(unsigned int *)(v51 + 84)];
              if ( v46 == v45 )
              {
                v68 = *(_QWORD **)(v51 + 64);
              }
              else
              {
                do
                {
                  if ( *(_QWORD *)(v50 + 40) == *v46 )
                    break;
                  ++v46;
                }
                while ( v45 != v46 );
                v68 = v45;
              }
            }
            else
            {
              v73 = *(_QWORD *)(v50 + 40);
              v45 = &v52[*(unsigned int *)(v51 + 80)];
              v46 = sub_16CC9F0(v51 + 56, v73);
              if ( v73 == *v46 )
              {
                v53 = *(_QWORD *)(v51 + 72);
                if ( v53 == *(_QWORD *)(v51 + 64) )
                  v54 = *(unsigned int *)(v51 + 84);
                else
                  v54 = *(unsigned int *)(v51 + 80);
                v68 = (_QWORD *)(v53 + 8 * v54);
              }
              else
              {
                v47 = *(_QWORD *)(v51 + 72);
                if ( v47 != *(_QWORD *)(v51 + 64) )
                {
                  v46 = (_QWORD *)(v47 + 8LL * *(unsigned int *)(v51 + 80));
                  goto LABEL_45;
                }
                v46 = (_QWORD *)(v47 + 8LL * *(unsigned int *)(v51 + 84));
                v68 = v46;
              }
            }
            while ( v68 != v46 && *v46 >= 0xFFFFFFFFFFFFFFFELL )
              ++v46;
LABEL_45:
            if ( v45 != v46 )
            {
              v48 = *v44;
              v49 = *(unsigned int *)(*v44 + 8);
              if ( (unsigned int)v49 >= *(_DWORD *)(*v44 + 12) )
              {
                sub_16CD150(*v44, (const void *)(v48 + 16), 0, 8, v15, a6);
                v49 = *(unsigned int *)(v48 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v48 + 8 * v49) = v50;
              ++*(_DWORD *)(v48 + 8);
            }
            goto LABEL_49;
          }
LABEL_13:
          v9 = *v6;
          result = *(unsigned int *)(*v6 + 8);
          if ( !(_DWORD)result )
            goto LABEL_14;
          continue;
        }
        break;
      }
      if ( !a3 )
        goto LABEL_13;
      v31 = sub_13FCB50(v30);
      v32 = *(_BYTE *)(v11 + 23) & 0x40;
      v33 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      if ( v33 )
      {
        a6 = v11 - 24 * v33;
        v34 = 24LL * *(unsigned int *)(v11 + 56) + 8;
        v35 = 0;
        while ( 1 )
        {
          v36 = v11 - 24LL * v33;
          if ( v32 )
            v36 = *(_QWORD *)(v11 - 8);
          if ( v31 == *(_QWORD *)(v36 + v34) )
            break;
          ++v35;
          v34 += 8;
          if ( v33 == (_DWORD)v35 )
            goto LABEL_81;
        }
        v37 = 24 * v35;
        if ( v32 )
        {
LABEL_30:
          v38 = *(_QWORD *)(*(_QWORD *)(v11 - 8) + v37);
          if ( !v38 )
            goto LABEL_83;
          goto LABEL_31;
        }
      }
      else
      {
LABEL_81:
        v37 = 0x17FFFFFFE8LL;
        if ( v32 )
          goto LABEL_30;
      }
      v38 = *(_QWORD *)(v11 - 24LL * v33 + v37);
      if ( !v38 )
LABEL_83:
        BUG();
LABEL_31:
      if ( *(_BYTE *)(v38 + 16) <= 0x17u || !sub_1377F70(*(_QWORD *)v6[2] + 56LL, *(_QWORD *)(v38 + 40)) )
        goto LABEL_13;
      v40 = v6[3];
      v41 = *(unsigned int *)(v40 + 8);
      if ( (unsigned int)v41 >= *(_DWORD *)(v40 + 12) )
      {
        sub_16CD150(v6[3], (const void *)(v40 + 16), 0, 8, v39, a6);
        v41 = *(unsigned int *)(v40 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v40 + 8 * v41) = v38;
      ++*(_DWORD *)(v40 + 8);
      v9 = *v6;
      result = *(unsigned int *)(*v6 + 8);
      if ( (_DWORD)result )
        continue;
      break;
    }
LABEL_14:
    v25 = v6[3];
    v26 = *(_DWORD *)(v25 + 8);
    if ( v26 )
    {
      v27 = v26;
      v28 = *(const void **)v25;
      v29 = 8LL * v26;
      if ( v26 > (unsigned __int64)*(unsigned int *)(v9 + 12) )
      {
        sub_16CD150(v9, (const void *)(v9 + 16), v26, 8, v29, a6);
        result = *(unsigned int *)(v9 + 8);
        v29 = 8 * v27;
      }
      memcpy((void *)(*(_QWORD *)v9 + 8 * result), v28, v29);
      *(_DWORD *)(v9 + 8) += v27;
      --a3;
      *(_DWORD *)(v6[3] + 8) = 0;
      continue;
    }
    return result;
  }
}
