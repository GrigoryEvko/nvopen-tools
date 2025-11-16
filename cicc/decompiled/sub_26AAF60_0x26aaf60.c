// Function: sub_26AAF60
// Address: 0x26aaf60
//
__int64 __fastcall sub_26AAF60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v14; // rbx
  char *v15; // r12
  char *v16; // r14
  __int64 v17; // r9
  __int64 v18; // r8
  _BYTE *v19; // r10
  int v20; // r11d
  unsigned int v21; // edx
  _BYTE *v22; // rax
  char v23; // cl
  unsigned int v24; // esi
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  unsigned int v28; // eax
  char v29; // cl
  int v30; // eax
  char v31; // r15
  __int64 v32; // rax
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  unsigned int v36; // ecx
  char v37; // dl
  __int64 v38; // [rsp+8h] [rbp-68h]
  unsigned __int64 v39; // [rsp+10h] [rbp-60h]
  unsigned __int64 v40[2]; // [rsp+28h] [rbp-48h] BYREF
  _BYTE v41[56]; // [rsp+38h] [rbp-38h] BYREF

  v7 = *(_QWORD *)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v40[0] = (unsigned __int64)v41;
  v40[1] = 0;
  if ( v8 )
    sub_266E590((__int64)v40, (char **)(a2 + 8), a3, a4, a5, a6);
  v9 = sub_B43CB0(v7);
  v10 = *a1;
  v11 = v9;
  v39 = v9 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v12 = sub_26A73D0(v10, v39, 0, 0, 2, 1);
  if ( !v12 || !*(_BYTE *)(v12 + 401) )
  {
    *(_BYTE *)(a1[2] + 401) = *(_BYTE *)(a1[2] + 400);
    goto LABEL_6;
  }
  v14 = a1[2];
  if ( v11 == *(_QWORD *)(a1[1] + 120) )
  {
    *(_BYTE *)(v14 + 401) = *(_BYTE *)(v14 + 400);
    goto LABEL_6;
  }
  v15 = *(char **)(v12 + 440);
  v38 = v14 + 408;
  v16 = &v15[*(_QWORD *)(v12 + 448)];
  if ( v15 != v16 )
  {
    while ( 1 )
    {
      v24 = *(_DWORD *)(v14 + 432);
      if ( !v24 )
        break;
      v17 = v24 - 1;
      v18 = *(_QWORD *)(v14 + 416);
      v19 = 0;
      v20 = 1;
      v21 = v17 & (37 * (unsigned __int8)*v15);
      v22 = (_BYTE *)(v18 + v21);
      v23 = *v22;
      if ( *v15 != *v22 )
      {
        while ( v23 != -1 )
        {
          if ( v23 != -2 || v19 )
            v22 = v19;
          v21 = v17 & (v20 + v21);
          v23 = *(_BYTE *)(v18 + v21);
          if ( *v15 == v23 )
            goto LABEL_13;
          ++v20;
          v19 = v22;
          v22 = (_BYTE *)(v18 + v21);
        }
        if ( !v19 )
          v19 = v22;
        v30 = *(_DWORD *)(v14 + 424);
        ++*(_QWORD *)(v14 + 408);
        if ( 4 * (v30 + 1) < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(v14 + 428) - (v30 + 1) <= v24 >> 3 )
          {
            sub_26AADA0(v38, v24);
            v33 = *(_DWORD *)(v14 + 432);
            if ( !v33 )
            {
LABEL_52:
              ++*(_DWORD *)(v14 + 424);
              BUG();
            }
            v34 = v33 - 1;
            v35 = *(_QWORD *)(v14 + 416);
            v18 = 0;
            v17 = 1;
            v36 = v34 & (37 * (unsigned __int8)*v15);
            v19 = (_BYTE *)(v35 + v36);
            v37 = *v19;
            if ( *v19 != *v15 )
            {
              while ( v37 != -1 )
              {
                if ( v37 == -2 && !v18 )
                  v18 = (__int64)v19;
                v36 = v34 & (v17 + v36);
                v19 = (_BYTE *)(v35 + v36);
                v37 = *v19;
                if ( *v15 == *v19 )
                  goto LABEL_33;
                v17 = (unsigned int)(v17 + 1);
              }
LABEL_20:
              if ( v18 )
                v19 = (_BYTE *)v18;
            }
          }
LABEL_33:
          ++*(_DWORD *)(v14 + 424);
          if ( *v19 != 0xFF )
            --*(_DWORD *)(v14 + 428);
          v31 = *v15;
          *v19 = *v15;
          v32 = *(_QWORD *)(v14 + 448);
          if ( (unsigned __int64)(v32 + 1) > *(_QWORD *)(v14 + 456) )
          {
            sub_C8D290(v14 + 440, (const void *)(v14 + 464), v32 + 1, 1u, v18, v17);
            v32 = *(_QWORD *)(v14 + 448);
          }
          *(_BYTE *)(*(_QWORD *)(v14 + 440) + v32) = v31;
          ++*(_QWORD *)(v14 + 448);
          goto LABEL_13;
        }
LABEL_16:
        sub_26AADA0(v38, 2 * v24);
        v25 = *(_DWORD *)(v14 + 432);
        if ( !v25 )
          goto LABEL_52;
        v26 = v25 - 1;
        v27 = *(_QWORD *)(v14 + 416);
        v28 = (v25 - 1) & (37 * (unsigned __int8)*v15);
        v19 = (_BYTE *)(v27 + (v26 & (37 * (unsigned int)(unsigned __int8)*v15)));
        v29 = *v19;
        if ( *v15 != *v19 )
        {
          v17 = 1;
          v18 = 0;
          while ( v29 != -1 )
          {
            if ( v29 == -2 && !v18 )
              v18 = (__int64)v19;
            v28 = v26 & (v17 + v28);
            v19 = (_BYTE *)(v27 + v28);
            v29 = *v19;
            if ( *v15 == *v19 )
              goto LABEL_33;
            v17 = (unsigned int)(v17 + 1);
          }
          goto LABEL_20;
        }
        goto LABEL_33;
      }
LABEL_13:
      if ( v16 == ++v15 )
        goto LABEL_6;
    }
    ++*(_QWORD *)(v14 + 408);
    goto LABEL_16;
  }
LABEL_6:
  if ( (_BYTE *)v40[0] != v41 )
    _libc_free(v40[0]);
  return 1;
}
