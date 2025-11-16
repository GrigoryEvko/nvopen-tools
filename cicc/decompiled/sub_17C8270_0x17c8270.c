// Function: sub_17C8270
// Address: 0x17c8270
//
__int64 __fastcall sub_17C8270(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 *v6; // r15
  __int64 *v7; // rbx
  unsigned int v8; // ebx
  unsigned int v11; // r14d
  _QWORD *v12; // rbx
  __int64 v13; // rax
  int v14; // esi
  __int64 v15; // r9
  int v16; // esi
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // r11
  __int64 v20; // r13
  unsigned int v21; // eax
  unsigned int v22; // r12d
  __int64 v23; // r15
  unsigned int v24; // esi
  __int64 v25; // r9
  __int64 v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // r8
  unsigned int v29; // esi
  int v30; // eax
  int v31; // ecx
  __int64 *v32; // rcx
  int v33; // ecx
  int v34; // edi
  int v35; // eax
  int v36; // r10d
  __int64 v37; // rcx
  unsigned int v38; // esi
  __int64 v39; // r9
  int v40; // r11d
  __int64 *v41; // r8
  int v42; // eax
  int v43; // r10d
  __int64 v44; // rcx
  int v45; // r11d
  unsigned int v46; // esi
  __int64 v47; // r9
  __int64 *v48; // [rsp+8h] [rbp-F8h]
  _QWORD *v49; // [rsp+18h] [rbp-E8h]
  __int64 *v50; // [rsp+28h] [rbp-D8h]
  __int64 *v51; // [rsp+28h] [rbp-D8h]
  int v52; // [rsp+28h] [rbp-D8h]
  __int64 *v53; // [rsp+28h] [rbp-D8h]
  _BYTE *v54; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+38h] [rbp-C8h]
  _BYTE v56[64]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE *v57; // [rsp+80h] [rbp-80h] BYREF
  __int64 v58; // [rsp+88h] [rbp-78h]
  _BYTE v59[112]; // [rsp+90h] [rbp-70h] BYREF

  v55 = 0x800000000LL;
  v54 = v56;
  sub_13F9EC0(a2, (__int64)&v54);
  v2 = v54;
  v3 = 8LL * (unsigned int)v55;
  v50 = (__int64 *)&v54[v3];
  v4 = v3 >> 3;
  v5 = v3 >> 5;
  if ( !v5 )
  {
    v6 = (__int64 *)v54;
LABEL_13:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          goto LABEL_17;
        goto LABEL_16;
      }
      if ( *(_BYTE *)(sub_157EBA0(*v6) + 16) == 34 )
        goto LABEL_8;
      ++v6;
    }
    if ( *(_BYTE *)(sub_157EBA0(*v6) + 16) == 34 )
      goto LABEL_8;
    ++v6;
LABEL_16:
    if ( *(_BYTE *)(sub_157EBA0(*v6) + 16) != 34 )
      goto LABEL_17;
    goto LABEL_8;
  }
  v6 = (__int64 *)v54;
  v7 = (__int64 *)&v54[32 * v5];
  while ( 1 )
  {
    if ( *(_BYTE *)(sub_157EBA0(*v6) + 16) == 34 )
      goto LABEL_8;
    if ( *(_BYTE *)(sub_157EBA0(v6[1]) + 16) == 34 )
    {
      ++v6;
      goto LABEL_8;
    }
    if ( *(_BYTE *)(sub_157EBA0(v6[2]) + 16) == 34 )
    {
      v6 += 2;
      goto LABEL_8;
    }
    if ( *(_BYTE *)(sub_157EBA0(v6[3]) + 16) == 34 )
      break;
    v6 += 4;
    if ( v7 == v6 )
    {
      v4 = v50 - v6;
      goto LABEL_13;
    }
  }
  v6 += 3;
LABEL_8:
  v8 = 0;
  if ( v50 != v6 )
    goto LABEL_9;
LABEL_17:
  if ( (unsigned __int8)sub_13FC370(a2) && sub_13FC520(a2) )
  {
    v58 = 0x800000000LL;
    v57 = v59;
    sub_13F9CA0(a2, (__int64)&v57);
    v8 = dword_4FA36E0;
    if ( (_DWORD)v58 == 1 )
      goto LABEL_36;
    if ( dword_4FA3520 < (unsigned int)v58 )
    {
      v8 = 0;
    }
    else
    {
      v8 = dword_4FA36E0;
      if ( !byte_4FA3440 )
      {
        v49 = &v54[8 * (unsigned int)v55];
        if ( v54 != (_BYTE *)v49 )
        {
          v11 = dword_4FA36E0;
          v12 = v54;
          while ( 1 )
          {
            v13 = a1[22];
            v14 = *(_DWORD *)(v13 + 24);
            if ( v14 )
            {
              v15 = *(_QWORD *)(v13 + 8);
              v16 = v14 - 1;
              v17 = v16 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
              v18 = (__int64 *)(v15 + 16LL * v17);
              v19 = *v18;
              if ( *v12 != *v18 )
              {
                v30 = 1;
                while ( v19 != -8 )
                {
                  v31 = v30 + 1;
                  v17 = v16 & (v30 + v17);
                  v18 = (__int64 *)(v15 + 16LL * v17);
                  v19 = *v18;
                  if ( *v12 == *v18 )
                    goto LABEL_26;
                  v30 = v31;
                }
                goto LABEL_34;
              }
LABEL_26:
              v20 = v18[1];
              if ( v20 )
                break;
            }
LABEL_34:
            if ( v49 == ++v12 )
            {
              v8 = v11;
              goto LABEL_36;
            }
          }
          v51 = a1;
          v21 = sub_17C8270(a1, v18[1]);
          a1 = v51;
          v22 = v21;
          v23 = *v51;
          v24 = *(_DWORD *)(*v51 + 24);
          if ( v24 )
          {
            v25 = *(_QWORD *)(v23 + 8);
            v26 = (v24 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
            v27 = (__int64 *)(v25 + 152 * v26);
            v28 = *v27;
            if ( v20 == *v27 )
            {
LABEL_29:
              v29 = *((_DWORD *)v27 + 4);
              if ( v22 < v29 )
                v22 = *((_DWORD *)v27 + 4);
              v22 -= v29;
LABEL_32:
              if ( v11 > v22 )
                v11 = v22;
              goto LABEL_34;
            }
            v52 = 1;
            v32 = 0;
            while ( v28 != -8 )
            {
              if ( !v32 && v28 == -16 )
                v32 = v27;
              LODWORD(v26) = (v24 - 1) & (v52 + v26);
              v27 = (__int64 *)(v25 + 152LL * (unsigned int)v26);
              v28 = *v27;
              if ( v20 == *v27 )
                goto LABEL_29;
              ++v52;
            }
            if ( v32 )
              v27 = v32;
            v33 = *(_DWORD *)(v23 + 16);
            ++*(_QWORD *)v23;
            v34 = v33 + 1;
            if ( 4 * (v33 + 1) < 3 * v24 )
            {
              if ( v24 - *(_DWORD *)(v23 + 20) - v34 > v24 >> 3 )
                goto LABEL_59;
              v48 = a1;
              sub_17C7F60(v23, v24);
              v42 = *(_DWORD *)(v23 + 24);
              if ( !v42 )
              {
LABEL_88:
                ++*(_DWORD *)(v23 + 16);
                BUG();
              }
              v43 = v42 - 1;
              v44 = *(_QWORD *)(v23 + 8);
              v41 = 0;
              v45 = 1;
              v46 = (v42 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
              v34 = *(_DWORD *)(v23 + 16) + 1;
              a1 = v48;
              v27 = (__int64 *)(v44 + 152LL * v46);
              v47 = *v27;
              if ( *v27 == v20 )
                goto LABEL_59;
              while ( v47 != -8 )
              {
                if ( !v41 && v47 == -16 )
                  v41 = v27;
                v46 = v43 & (v45 + v46);
                v27 = (__int64 *)(v44 + 152LL * v46);
                v47 = *v27;
                if ( v20 == *v27 )
                  goto LABEL_59;
                ++v45;
              }
              goto LABEL_67;
            }
          }
          else
          {
            ++*(_QWORD *)v23;
          }
          v53 = a1;
          sub_17C7F60(v23, 2 * v24);
          v35 = *(_DWORD *)(v23 + 24);
          if ( !v35 )
            goto LABEL_88;
          v36 = v35 - 1;
          v37 = *(_QWORD *)(v23 + 8);
          v38 = (v35 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v34 = *(_DWORD *)(v23 + 16) + 1;
          a1 = v53;
          v27 = (__int64 *)(v37 + 152LL * v38);
          v39 = *v27;
          if ( v20 == *v27 )
            goto LABEL_59;
          v40 = 1;
          v41 = 0;
          while ( v39 != -8 )
          {
            if ( !v41 && v39 == -16 )
              v41 = v27;
            v38 = v36 & (v40 + v38);
            v27 = (__int64 *)(v37 + 152LL * v38);
            v39 = *v27;
            if ( v20 == *v27 )
              goto LABEL_59;
            ++v40;
          }
LABEL_67:
          if ( v41 )
            v27 = v41;
LABEL_59:
          *(_DWORD *)(v23 + 16) = v34;
          if ( *v27 != -8 )
            --*(_DWORD *)(v23 + 20);
          *v27 = v20;
          v27[1] = (__int64)(v27 + 3);
          v27[2] = 0x800000000LL;
          goto LABEL_32;
        }
      }
    }
LABEL_36:
    if ( v57 != v59 )
      _libc_free((unsigned __int64)v57);
    v2 = v54;
  }
  else
  {
    v2 = v54;
    v8 = 0;
  }
LABEL_9:
  if ( v2 != v56 )
    _libc_free((unsigned __int64)v2);
  return v8;
}
