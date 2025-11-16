// Function: sub_26A4B40
// Address: 0x26a4b40
//
void __fastcall sub_26A4B40(__int64 *a1, char a2)
{
  __int64 v2; // r8
  __int64 *v3; // r13
  __int64 i; // rax
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // r12
  __int64 v23; // r14
  unsigned __int64 v24; // r15
  __int64 v25; // rbx
  unsigned __int8 *v26; // rdi
  int v27; // eax
  unsigned __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  int v33; // edx
  int v34; // ecx
  int v35; // r8d
  unsigned int v36; // edx
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // r15
  __int64 v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r13
  __int64 v53; // [rsp-B8h] [rbp-B8h]
  __int64 v54; // [rsp-B8h] [rbp-B8h]
  __int64 v55; // [rsp-A8h] [rbp-A8h]
  __int64 *v56; // [rsp-A0h] [rbp-A0h]
  __int64 *v57; // [rsp-A0h] [rbp-A0h]
  __int64 v58; // [rsp-98h] [rbp-98h]
  __int64 v59; // [rsp-90h] [rbp-90h]
  __int64 v60; // [rsp-90h] [rbp-90h]
  __int64 v61; // [rsp-90h] [rbp-90h]
  __int64 *v62; // [rsp-80h] [rbp-80h] BYREF
  unsigned __int64 v63; // [rsp-78h] [rbp-78h]
  __int64 v64; // [rsp-70h] [rbp-70h]
  _BYTE *v65; // [rsp-68h] [rbp-68h]
  __int64 v66; // [rsp-60h] [rbp-60h]
  _BYTE v67[88]; // [rsp-58h] [rbp-58h] BYREF

  v2 = a1[5];
  if ( *(_DWORD *)(v2 + 8) )
  {
    v3 = a1;
    if ( !a2 )
      goto LABEL_3;
    v38 = a1[9];
    v62 = a1;
    sub_26807A0(v38 + 28312, v2, (__int64 (__fastcall *)(__int64, _QWORD, __int64))sub_26AAC40, (__int64)&v62);
    sub_26A4990(a1, 186);
    sub_26A4990(a1, 185);
    sub_26A4990(a1, 15);
    v54 = a1[9];
    v39 = a1[5];
    v40 = *(__int64 **)v39;
    v61 = *(_QWORD *)v39 + 8LL * *(unsigned int *)(v39 + 8);
    if ( *(_QWORD *)v39 == v61 )
    {
LABEL_3:
      if ( !byte_4FF5228 )
        goto LABEL_25;
    }
    else
    {
      do
      {
        v41 = *v40;
        v65 = v67;
        v66 = 0x800000000LL;
        v42 = sub_267FA80(v54 + 5752, v41);
        v43 = *v42 + 8LL * *((unsigned int *)v42 + 2);
        if ( *v42 != v43 )
        {
          v57 = v40;
          v44 = *v42;
          do
          {
            while ( 1 )
            {
              v45 = *(_QWORD *)(*(_QWORD *)v44 + 24LL);
              if ( *(_BYTE *)v45 == 85 && *(_QWORD *)v44 == v45 - 32 )
              {
                if ( *(char *)(v45 + 7) >= 0 )
                  goto LABEL_70;
                v46 = sub_BD2BC0(*(_QWORD *)(*(_QWORD *)v44 + 24LL));
                v48 = v46 + v47;
                v49 = 0;
                if ( *(char *)(v45 + 7) < 0 )
                  v49 = sub_BD2BC0(v45);
                if ( !(unsigned int)((v48 - v49) >> 4) )
                {
LABEL_70:
                  v50 = *(_QWORD *)(v54 + 5872);
                  if ( v50 )
                  {
                    v51 = *(_QWORD *)(v45 - 32);
                    if ( v51 )
                    {
                      if ( !*(_BYTE *)v51 && *(_QWORD *)(v51 + 24) == *(_QWORD *)(v45 + 80) && v50 == v51 )
                        break;
                    }
                  }
                }
              }
              v44 += 8;
              if ( v43 == v44 )
                goto LABEL_59;
            }
            v64 = 0;
            v44 += 8;
            v52 = a1[10];
            v63 = v45 & 0xFFFFFFFFFFFFFFFCLL | 1;
            nullsub_1518();
            sub_26A45D0(v52, v63, v64);
          }
          while ( v43 != v44 );
LABEL_59:
          v40 = v57;
        }
        ++v40;
      }
      while ( (__int64 *)v61 != v40 );
      v3 = a1;
      if ( !byte_4FF5228 )
      {
LABEL_25:
        if ( !sub_2674830(v3[4]) )
          return;
        v21 = v3[5];
        v22 = *(__int64 **)v21;
        v23 = *(_QWORD *)v21 + 8LL * *(unsigned int *)(v21 + 8);
        if ( v23 == *(_QWORD *)v21 )
          return;
        while ( 1 )
        {
LABEL_30:
          v24 = *v22;
          if ( !sub_B2FC80(*v22) )
          {
            if ( (*(_BYTE *)(v24 + 32) & 0xFu) - 7 <= 1 )
            {
              v25 = *(_QWORD *)(v24 + 16);
              if ( !v25 )
                goto LABEL_29;
              while ( 1 )
              {
                v26 = *(unsigned __int8 **)(v25 + 24);
                v27 = *v26;
                if ( (unsigned __int8)v27 <= 0x1Cu )
                  break;
                v28 = (unsigned int)(v27 - 34);
                if ( (unsigned __int8)v28 > 0x33u )
                  break;
                v29 = 0x8000000000041LL;
                if ( !_bittest64(&v29, v28) || (unsigned __int8 *)v25 != v26 - 32 )
                  break;
                v60 = v3[10];
                v30 = sub_B491C0((__int64)v26);
                v31 = *(_QWORD *)(v60 + 200);
                if ( *(_DWORD *)(v31 + 40) )
                {
                  v32 = *(_QWORD *)(v31 + 8);
                  v33 = *(_DWORD *)(v31 + 24);
                  if ( !v33 )
                    break;
                  v34 = v33 - 1;
                  v35 = 1;
                  v36 = (v33 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                  v37 = *(_QWORD *)(v32 + 8LL * v36);
                  if ( v37 != v30 )
                  {
                    while ( v37 != -4096 )
                    {
                      v36 = v34 & (v35 + v36);
                      v37 = *(_QWORD *)(v32 + 8LL * v36);
                      if ( v30 == v37 )
                        goto LABEL_40;
                      ++v35;
                    }
                    break;
                  }
                }
LABEL_40:
                v25 = *(_QWORD *)(v25 + 8);
                if ( !v25 )
                {
                  if ( (__int64 *)v23 != ++v22 )
                    goto LABEL_30;
                  return;
                }
              }
            }
            sub_26A1DB0(v3[10], v24);
          }
LABEL_29:
          if ( (__int64 *)v23 == ++v22 )
            return;
        }
      }
    }
    v58 = 0;
    for ( i = v3[9]; ; i = v3[9] )
    {
      v55 = i + 160LL * *(int *)(i + v58 + 34644) + 3512;
      v5 = v3[5];
      v6 = *(__int64 **)v5;
      v59 = *(_QWORD *)v5 + 8LL * *(unsigned int *)(v5 + 8);
      if ( *(_QWORD *)v5 != v59 )
      {
        do
        {
          v7 = *v6;
          v65 = v67;
          v66 = 0x800000000LL;
          v8 = sub_267FA80(v55, v7);
          v9 = *v8;
          v10 = *((unsigned int *)v8 + 2);
          if ( v9 != v9 + 8 * v10 )
          {
            v56 = v6;
            v11 = v9 + 8 * v10;
            v12 = v9;
            do
            {
              while ( 1 )
              {
                v13 = *(_QWORD *)(*(_QWORD *)v12 + 24LL);
                if ( *(_BYTE *)v13 == 85 && *(_QWORD *)v12 == v13 - 32 )
                {
                  if ( *(char *)(v13 + 7) >= 0 )
                    goto LABEL_71;
                  v14 = sub_BD2BC0(*(_QWORD *)(*(_QWORD *)v12 + 24LL));
                  v16 = v14 + v15;
                  v17 = 0;
                  if ( *(char *)(v13 + 7) < 0 )
                  {
                    v53 = v16;
                    v18 = sub_BD2BC0(v13);
                    v16 = v53;
                    v17 = v18;
                  }
                  if ( !(unsigned int)((v16 - v17) >> 4) )
                  {
LABEL_71:
                    v19 = *(_QWORD *)(v55 + 120);
                    if ( v19 )
                    {
                      v20 = *(_QWORD *)(v13 - 32);
                      if ( v20 )
                      {
                        if ( !*(_BYTE *)v20 && *(_QWORD *)(v20 + 24) == *(_QWORD *)(v13 + 80) && v19 == v20 )
                          break;
                      }
                    }
                  }
                }
                v12 += 8;
                if ( v11 == v12 )
                  goto LABEL_21;
              }
              v64 = 0;
              v12 += 8;
              v63 = v13 & 0xFFFFFFFFFFFFFFFCLL;
              nullsub_1518();
              sub_26A2E60(v3[10], v63, v64, 0, 2);
            }
            while ( v11 != v12 );
LABEL_21:
            v6 = v56;
          }
          ++v6;
        }
        while ( (__int64 *)v59 != v6 );
      }
      v58 += 72;
      if ( v58 == 288 )
        break;
    }
    goto LABEL_25;
  }
}
