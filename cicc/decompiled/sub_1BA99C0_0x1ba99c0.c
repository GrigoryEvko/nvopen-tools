// Function: sub_1BA99C0
// Address: 0x1ba99c0
//
__int64 __fastcall sub_1BA99C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v7; // rax
  __int64 v8; // r9
  char v9; // al
  __int64 v10; // r9
  char v11; // r10
  __int64 *v12; // rax
  int v13; // esi
  int v14; // edx
  unsigned int v15; // esi
  __int64 v16; // rdx
  __int64 v18; // r12
  char v19; // al
  __int64 *v20; // rdx
  __int64 v21; // rdi
  unsigned __int64 v22; // rcx
  __int64 *v23; // r9
  __int64 v24; // r12
  __int64 v25; // rax
  int v26; // r8d
  __int64 v27; // r10
  __int64 v28; // rbx
  __int64 v29; // r11
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 **v37; // r14
  __int64 v38; // r15
  __int64 v39; // r13
  __int64 v40; // rbx
  const void *v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  int v44; // eax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // [rsp+0h] [rbp-90h]
  __int64 *v48; // [rsp+10h] [rbp-80h]
  __int64 *v49; // [rsp+10h] [rbp-80h]
  __int64 v50; // [rsp+10h] [rbp-80h]
  __int64 v51; // [rsp+18h] [rbp-78h]
  unsigned __int64 v52; // [rsp+18h] [rbp-78h]
  __int64 *v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  __int64 *v55; // [rsp+18h] [rbp-78h]
  __int64 v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+18h] [rbp-78h]
  __int64 v59; // [rsp+28h] [rbp-68h]
  __int64 v60; // [rsp+28h] [rbp-68h]
  __int64 v61; // [rsp+28h] [rbp-68h]
  __int64 v62; // [rsp+28h] [rbp-68h]
  __int64 v63; // [rsp+38h] [rbp-58h] BYREF
  __int64 v64; // [rsp+40h] [rbp-50h] BYREF
  __int64 v65; // [rsp+48h] [rbp-48h]
  __int64 *v66[2]; // [rsp+50h] [rbp-40h] BYREF
  char v67; // [rsp+60h] [rbp-30h] BYREF

  v4 = a1;
  v5 = a1 + 48;
  v64 = a2;
  v65 = a3;
  if ( (unsigned __int8)sub_1B997F0(a1 + 48, &v64, v66)
    && v66[0] != (__int64 *)(*(_QWORD *)(a1 + 56) + 24LL * *(unsigned int *)(a1 + 72)) )
  {
    return v66[0][2];
  }
  v51 = sub_1BA9430(a1, a2, (__int64)a4);
  v7 = sub_157EBA0(a2);
  if ( *(_BYTE *)(v7 + 16) != 26 )
    BUG();
  v8 = v51;
  if ( (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) == 3 )
  {
    v48 = (__int64 *)v51;
    v52 = v7;
    v18 = *a4;
    v63 = *(_QWORD *)(v7 - 72);
    v19 = sub_1BA0BD0(v18 + 280, &v63, v66);
    v20 = v66[0];
    v21 = v18 + 280;
    v22 = v52;
    v23 = v48;
    if ( v19 )
    {
      v24 = v66[0][1];
    }
    else
    {
      v43 = *(_DWORD *)(v18 + 304);
      v44 = *(_DWORD *)(v18 + 296);
      ++*(_QWORD *)(v18 + 280);
      v45 = v44 + 1;
      if ( 4 * v45 >= 3 * v43 )
      {
        sub_1BA21E0(v21, 2 * v43);
        sub_1BA0BD0(v21, &v63, v66);
        v20 = v66[0];
        v22 = v52;
        v23 = v48;
        v45 = *(_DWORD *)(v18 + 296) + 1;
      }
      else if ( v43 - *(_DWORD *)(v18 + 300) - v45 <= v43 >> 3 )
      {
        sub_1BA21E0(v21, v43);
        sub_1BA0BD0(v21, &v63, v66);
        v20 = v66[0];
        v23 = v48;
        v22 = v52;
        v45 = *(_DWORD *)(v18 + 296) + 1;
      }
      *(_DWORD *)(v18 + 296) = v45;
      if ( *v20 != -8 )
        --*(_DWORD *)(v18 + 300);
      v46 = v63;
      v20[1] = 0;
      v24 = 0;
      *v20 = v46;
    }
    if ( *(_QWORD *)(v22 - 24) != a3 )
    {
      v53 = v23;
      v60 = *(_QWORD *)(v4 + 40);
      v25 = sub_22077B0(120);
      v27 = v60;
      v23 = v53;
      v28 = v25;
      if ( v25 )
      {
        *(_BYTE *)(v25 + 40) = 2;
        v29 = v25 + 40;
        *(_QWORD *)(v25 + 48) = v25 + 64;
        *(_QWORD *)(v25 + 56) = 0x100000000LL;
        *(_QWORD *)(v25 + 80) = v25 + 96;
        *(_QWORD *)(v25 + 72) = 0;
        *(_QWORD *)(v25 + 96) = v24;
        *(_QWORD *)(v25 + 88) = 0x200000001LL;
        v30 = *(unsigned int *)(v24 + 16);
        if ( (unsigned int)v30 >= *(_DWORD *)(v24 + 20) )
        {
          v49 = v53;
          v54 = v29;
          sub_16CD150(v24 + 8, (const void *)(v24 + 24), 0, 8, v26, (int)v23);
          v30 = *(unsigned int *)(v24 + 16);
          v23 = v49;
          v29 = v54;
          v27 = v60;
        }
        *(_QWORD *)(*(_QWORD *)(v24 + 8) + 8 * v30) = v29;
        ++*(_DWORD *)(v24 + 16);
        *(_QWORD *)(v28 + 8) = 0;
        *(_QWORD *)(v28 + 16) = 0;
        *(_BYTE *)(v28 + 24) = 2;
        *(_QWORD *)(v28 + 32) = 0;
        *(_QWORD *)v28 = &unk_49F7160;
        *(_BYTE *)(v28 + 112) = 66;
        if ( *(_QWORD *)v27 )
        {
          v55 = v23;
          v61 = v29;
          sub_1B91070(*(_QWORD *)v27, (_QWORD *)v28, *(unsigned __int64 **)(v27 + 8));
          v29 = v61;
          v23 = v55;
        }
        v24 = v29;
      }
      else
      {
        v24 = *(_QWORD *)v60;
        if ( *(_QWORD *)v60 )
        {
          v24 = 0;
          sub_1B91070(*(_QWORD *)v60, 0, *(unsigned __int64 **)(v60 + 8));
          v23 = v53;
        }
      }
    }
    if ( v23 )
    {
      v31 = *(_QWORD *)(v4 + 40);
      v66[0] = (__int64 *)v24;
      v66[1] = v23;
      v62 = v31;
      v32 = sub_22077B0(120);
      if ( v32 )
      {
        *(_BYTE *)(v32 + 40) = 2;
        v34 = v32 + 40;
        v35 = 0;
        *(_QWORD *)(v32 + 48) = v32 + 64;
        *(_QWORD *)(v32 + 56) = 0x100000000LL;
        v36 = v32 + 96;
        *(_QWORD *)(v32 + 72) = 0;
        *(_QWORD *)(v32 + 80) = v32 + 96;
        *(_QWORD *)(v32 + 88) = 0x200000000LL;
        v56 = v5;
        v37 = v66;
        v50 = v4;
        v38 = v32 + 40;
        v39 = v32;
        v40 = v24;
        v41 = (const void *)(v32 + 96);
        v47 = v32 + 80;
        while ( 1 )
        {
          *(_QWORD *)(v36 + 8 * v35) = v40;
          ++*(_DWORD *)(v39 + 88);
          v42 = *(unsigned int *)(v40 + 16);
          if ( (unsigned int)v42 >= *(_DWORD *)(v40 + 20) )
          {
            sub_16CD150(v40 + 8, (const void *)(v40 + 24), 0, 8, v33, v34);
            v42 = *(unsigned int *)(v40 + 16);
          }
          ++v37;
          *(_QWORD *)(*(_QWORD *)(v40 + 8) + 8 * v42) = v38;
          ++*(_DWORD *)(v40 + 16);
          if ( v37 == (__int64 **)&v67 )
            break;
          v40 = (__int64)*v37;
          v35 = *(unsigned int *)(v39 + 88);
          if ( (unsigned int)v35 >= *(_DWORD *)(v39 + 92) )
          {
            sub_16CD150(v47, v41, 0, 8, v33, v34);
            v35 = *(unsigned int *)(v39 + 88);
          }
          v36 = *(_QWORD *)(v39 + 80);
        }
        v8 = v38;
        v5 = v56;
        *(_QWORD *)(v39 + 8) = 0;
        v4 = v50;
        *(_QWORD *)(v39 + 16) = 0;
        *(_QWORD *)v39 = &unk_49F7160;
        *(_BYTE *)(v39 + 24) = 2;
        *(_QWORD *)(v39 + 32) = 0;
        *(_BYTE *)(v39 + 112) = 26;
        if ( *(_QWORD *)v62 )
        {
          v57 = v8;
          sub_1B91070(*(_QWORD *)v62, (_QWORD *)v39, *(unsigned __int64 **)(v62 + 8));
          v8 = v57;
        }
      }
      else
      {
        v8 = *(_QWORD *)v62;
        if ( *(_QWORD *)v62 )
        {
          sub_1B91070(*(_QWORD *)v62, 0, *(unsigned __int64 **)(v62 + 8));
          v8 = 0;
        }
      }
    }
    else
    {
      v8 = v24;
    }
  }
  v59 = v8;
  v9 = sub_1B997F0(v5, &v64, v66);
  v10 = v59;
  v11 = v9;
  v12 = v66[0];
  if ( !v11 )
  {
    v13 = *(_DWORD *)(v4 + 64);
    ++*(_QWORD *)(v4 + 48);
    v14 = v13 + 1;
    v15 = *(_DWORD *)(v4 + 72);
    if ( 4 * v14 >= 3 * v15 )
    {
      v15 *= 2;
    }
    else if ( v15 - *(_DWORD *)(v4 + 68) - v14 > v15 >> 3 )
    {
      goto LABEL_7;
    }
    sub_1BA8EF0(v5, v15);
    sub_1B997F0(v5, &v64, v66);
    v12 = v66[0];
    v10 = v59;
    v14 = *(_DWORD *)(v4 + 64) + 1;
LABEL_7:
    *(_DWORD *)(v4 + 64) = v14;
    if ( *v12 != -8 || v12[1] != -8 )
      --*(_DWORD *)(v4 + 68);
    *v12 = v64;
    v16 = v65;
    v12[2] = 0;
    v12[1] = v16;
  }
  v12[2] = v10;
  return v10;
}
