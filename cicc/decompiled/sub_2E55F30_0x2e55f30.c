// Function: sub_2E55F30
// Address: 0x2e55f30
//
__int64 **__fastcall sub_2E55F30(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 **result; // rax
  __int64 v4; // r12
  __int64 *v5; // r9
  __int64 v6; // r15
  unsigned int v7; // r14d
  __int64 v8; // r15
  int v9; // esi
  __int64 *v10; // r8
  int v11; // eax
  __int64 v12; // rax
  _QWORD *v13; // rcx
  unsigned int v14; // r14d
  int v15; // eax
  __int64 *v16; // r9
  __int64 *v17; // r8
  unsigned int j; // r10d
  __int64 v19; // rdi
  __int64 *v20; // rcx
  __int64 v21; // r11
  char v22; // al
  unsigned int v23; // r14d
  int v24; // eax
  int v25; // r9d
  unsigned int i; // r8d
  __int64 v27; // rdi
  __int64 *v28; // rcx
  __int64 v29; // r10
  char v30; // al
  char v31; // al
  __int64 *v32; // rcx
  int v33; // eax
  __int64 v34; // rdi
  char v35; // al
  unsigned int v36; // r10d
  __int64 *v37; // [rsp+8h] [rbp-68h]
  __int64 *v38; // [rsp+8h] [rbp-68h]
  __int64 *v39; // [rsp+8h] [rbp-68h]
  __int64 *v40; // [rsp+10h] [rbp-60h]
  __int64 *v41; // [rsp+10h] [rbp-60h]
  __int64 *v42; // [rsp+10h] [rbp-60h]
  unsigned int v43; // [rsp+1Ch] [rbp-54h]
  int v44; // [rsp+1Ch] [rbp-54h]
  unsigned int v45; // [rsp+1Ch] [rbp-54h]
  int v46; // [rsp+20h] [rbp-50h]
  __int64 *v47; // [rsp+20h] [rbp-50h]
  int v48; // [rsp+20h] [rbp-50h]
  __int64 *v49; // [rsp+28h] [rbp-48h]
  __int64 *v50; // [rsp+28h] [rbp-48h]
  unsigned int v51; // [rsp+28h] [rbp-48h]
  __int64 *v52; // [rsp+28h] [rbp-48h]
  unsigned int v53; // [rsp+28h] [rbp-48h]
  __int64 *v54[7]; // [rsp+38h] [rbp-38h] BYREF

  *(_QWORD *)(*a1 + 136LL) = a1[1];
  v1 = (_QWORD *)a1[2];
  result = v54;
  if ( v1 )
  {
    while ( 1 )
    {
      v4 = *a1;
      v5 = v1 + 2;
      v6 = *(_QWORD *)(*a1 + 112LL);
      v7 = *(_DWORD *)(*a1 + 128LL);
      if ( !v1[1] )
      {
        if ( !v7 )
          goto LABEL_17;
        v23 = v7 - 1;
        v24 = sub_2E8E920(v1 + 2);
        v25 = 1;
        for ( i = v23 & v24; ; i = v23 & (v48 + v53) )
        {
          v27 = v1[2];
          v28 = (__int64 *)(v6 + 16LL * i);
          v29 = *v28;
          if ( (unsigned __int64)(v27 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v29 - 1) > 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v27 == v29 )
            {
LABEL_24:
              *v28 = -1;
              --*(_DWORD *)(v4 + 120);
              ++*(_DWORD *)(v4 + 124);
              v4 = *a1;
              goto LABEL_17;
            }
          }
          else
          {
            v44 = v25;
            v47 = (__int64 *)(v6 + 16LL * i);
            v51 = i;
            v30 = sub_2E88AF0(v27, *v28, 3);
            i = v51;
            v28 = v47;
            v25 = v44;
            if ( v30 )
              goto LABEL_24;
            v29 = *v47;
          }
          v48 = v25;
          v53 = i;
          if ( sub_2E4F140(v29, 0) )
            goto LABEL_16;
          v25 = v48 + 1;
        }
      }
      if ( !v7 )
        break;
      v14 = v7 - 1;
      v15 = sub_2E8E920(v1 + 2);
      v46 = 1;
      v16 = v1 + 2;
      v17 = 0;
      for ( j = v14 & v15; ; j = v14 & v36 )
      {
        v19 = v1[2];
        v20 = (__int64 *)(v6 + 16LL * j);
        v21 = *v20;
        if ( (unsigned __int64)(*v20 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v19 - 1) > 0xFFFFFFFFFFFFFFFDLL )
        {
          if ( v19 == v21 )
          {
LABEL_14:
            v13 = v20 + 1;
            goto LABEL_15;
          }
        }
        else
        {
          v37 = v16;
          v40 = (__int64 *)(v6 + 16LL * j);
          v43 = j;
          v50 = v17;
          v22 = sub_2E88AF0(v19, *v20, 3);
          v17 = v50;
          j = v43;
          v20 = v40;
          v16 = v37;
          if ( v22 )
            goto LABEL_14;
          v21 = *v40;
        }
        v38 = v16;
        v41 = v20;
        v45 = j;
        v52 = v17;
        v31 = sub_2E4F140(v21, 0);
        v10 = v52;
        v32 = v41;
        v5 = v38;
        if ( v31 )
          break;
        v34 = *v41;
        v42 = v38;
        v39 = v32;
        v35 = sub_2E4F140(v34, -1);
        v17 = v52;
        v16 = v42;
        if ( !v52 && v35 )
          v17 = v39;
        v36 = v46 + v45;
        ++v46;
      }
      v33 = *(_DWORD *)(v4 + 120);
      v8 = v4 + 104;
      v7 = *(_DWORD *)(v4 + 128);
      if ( !v52 )
        v10 = v41;
      ++*(_QWORD *)(v4 + 104);
      v11 = v33 + 1;
      v54[0] = v10;
      if ( 4 * v11 >= 3 * v7 )
        goto LABEL_5;
      if ( v7 - (v11 + *(_DWORD *)(v4 + 124)) <= v7 >> 3 )
      {
        v49 = v38;
        v9 = v7;
        goto LABEL_6;
      }
LABEL_7:
      *(_DWORD *)(v4 + 120) = v11;
      if ( *v10 )
        --*(_DWORD *)(v4 + 124);
      v12 = v1[2];
      v13 = v10 + 1;
      v10[1] = 0;
      *v10 = v12;
LABEL_15:
      *v13 = v1[1];
LABEL_16:
      v4 = *a1;
LABEL_17:
      a1[2] = *v1;
      result = *(__int64 ***)v4;
      *v1 = *(_QWORD *)v4;
      *(_QWORD *)v4 = v1;
      v1 = (_QWORD *)a1[2];
      if ( !v1 )
        return result;
    }
    ++*(_QWORD *)(v4 + 104);
    v8 = v4 + 104;
    v54[0] = 0;
LABEL_5:
    v49 = v5;
    v9 = 2 * v7;
LABEL_6:
    sub_2E55AD0(v8, v9);
    sub_2E513B0(v8, v49, v54);
    v10 = v54[0];
    v11 = *(_DWORD *)(v4 + 120) + 1;
    goto LABEL_7;
  }
  return result;
}
