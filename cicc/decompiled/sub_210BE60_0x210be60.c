// Function: sub_210BE60
// Address: 0x210be60
//
__int64 __fastcall sub_210BE60(_QWORD *a1)
{
  _QWORD *v1; // r13
  __int64 result; // rax
  __int64 v3; // r12
  int v4; // edx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned int v7; // eax
  int v8; // r8d
  int *v9; // rbx
  _QWORD *v10; // r14
  __int64 v11; // r15
  int v12; // r12d
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // r10d
  __int64 v16; // r13
  unsigned __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // r9
  __int64 v20; // r11
  unsigned int v21; // r12d
  __int64 v22; // rcx
  void (*v23)(); // rax
  __int64 v24; // r15
  __int64 *v25; // rax
  unsigned __int64 *v26; // r13
  unsigned __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  void (*v32)(); // rax
  __int64 v33; // r12
  __int64 v34; // rbx
  __int64 *v35; // rax
  unsigned __int64 *v36; // r15
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rdi
  _QWORD *v39; // rax
  _QWORD *v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdi
  int v46; // r9d
  __int64 v47; // r14
  __int64 **v48; // rsi
  __int64 v49; // rbx
  __int64 v50; // [rsp+8h] [rbp-78h]
  unsigned __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+20h] [rbp-60h]
  __int64 v54; // [rsp+20h] [rbp-60h]
  int v55; // [rsp+20h] [rbp-60h]
  int *v56; // [rsp+28h] [rbp-58h]
  int *v57; // [rsp+30h] [rbp-50h] BYREF
  __int64 v58; // [rsp+38h] [rbp-48h]
  _BYTE v59[64]; // [rsp+40h] [rbp-40h] BYREF

  v1 = a1;
  sub_210BC20(a1);
  while ( 1 )
  {
    result = (*(__int64 (__fastcall **)(_QWORD *))(*v1 + 48LL))(v1);
    v3 = result;
    if ( !result )
      return result;
    v4 = *(_DWORD *)(result + 112);
    v5 = v1[2];
    if ( v4 < 0 )
      v6 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 16LL * (v4 & 0x7FFFFFFF) + 8);
    else
      v6 = *(_QWORD *)(*(_QWORD *)(v5 + 272) + 8LL * (unsigned int)v4);
    if ( !v6 )
      goto LABEL_43;
    if ( (*(_BYTE *)(v6 + 4) & 8) != 0 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 32);
        if ( !v6 )
          break;
        if ( (*(_BYTE *)(v6 + 4) & 8) == 0 )
          goto LABEL_7;
      }
LABEL_43:
      v32 = *(void (**)())(*v1 + 64LL);
      if ( v32 != nullsub_737 )
      {
        ((void (__fastcall *)(_QWORD *, __int64))v32)(v1, v3);
        v4 = *(_DWORD *)(v3 + 112);
      }
      v33 = v1[4];
      v34 = 8LL * (v4 & 0x7FFFFFFF);
      v35 = (__int64 *)(v34 + *(_QWORD *)(v33 + 400));
      v36 = (unsigned __int64 *)*v35;
      if ( *v35 )
      {
        sub_1DB4CE0(*v35);
        v37 = v36[12];
        if ( v37 )
        {
          sub_210BA00(*(_QWORD *)(v37 + 16));
          j_j___libc_free_0(v37, 48);
        }
        v38 = v36[8];
        if ( (unsigned __int64 *)v38 != v36 + 10 )
          _libc_free(v38);
        if ( (unsigned __int64 *)*v36 != v36 + 2 )
          _libc_free(*v36);
        j_j___libc_free_0(v36, 120);
        v35 = (__int64 *)(v34 + *(_QWORD *)(v33 + 400));
      }
      *v35 = 0;
    }
    else
    {
LABEL_7:
      ++*(_DWORD *)(v1[5] + 256LL);
      v57 = (int *)v59;
      v58 = 0x400000000LL;
      v7 = (*(__int64 (__fastcall **)(_QWORD *, __int64, int **))(*v1 + 56LL))(v1, v3, &v57);
      if ( v7 == -1 )
      {
        v42 = *(unsigned int *)(v3 + 112);
        v43 = v1[2];
        if ( (int)v42 >= 0 )
        {
          v44 = *(_QWORD *)(*(_QWORD *)(v43 + 272) + 8 * v42);
          goto LABEL_66;
        }
        v44 = *(_QWORD *)(*(_QWORD *)(v43 + 24) + 16 * (v42 & 0x7FFFFFFF) + 8);
        if ( !v44 )
LABEL_76:
          sub_16BD130("ran out of registers during register allocation", 1u);
        while ( 1 )
        {
          v45 = *(_QWORD *)(v44 + 16);
          do
            v44 = *(_QWORD *)(v44 + 32);
          while ( v44 && v45 == *(_QWORD *)(v44 + 16) );
          if ( **(_WORD **)(v45 + 16) == 1 )
            break;
LABEL_66:
          if ( !v44 )
            goto LABEL_76;
        }
        sub_1E1A6B0(v45, "inline assembly requires more registers than available", 54);
        v46 = *(_DWORD *)(v3 + 112);
        v47 = v1[3];
        v48 = (__int64 **)(*(_QWORD *)(*(_QWORD *)(v1[2] + 24LL) + 16LL * (v46 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
        v49 = v1[6] + 24LL * *((unsigned __int16 *)*v48 + 12);
        if ( *((_DWORD *)v1 + 14) != *(_DWORD *)v49 )
        {
          sub_1ED7890((__int64)(v1 + 6), v48);
          v46 = *(_DWORD *)(v3 + 112);
        }
        sub_1F5BDB0(v47, v46, **(_WORD **)(v49 + 16));
        v31 = (unsigned __int64)v57;
        if ( v57 != (int *)v59 )
          goto LABEL_36;
      }
      else
      {
        if ( v7 )
          sub_2103010((_QWORD *)v1[5], v3, v7);
        v9 = v57;
        v56 = &v57[(unsigned int)v58];
        if ( v57 != v56 )
        {
          v10 = v1;
          while ( 1 )
          {
            while ( 1 )
            {
              v15 = *v9;
              v16 = v10[4];
              v17 = *(unsigned int *)(v16 + 408);
              v18 = *v9 & 0x7FFFFFFF;
              v19 = v18;
              v20 = 8LL * v18;
              if ( v18 >= (unsigned int)v17 )
                break;
              v11 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8LL * v18);
              if ( !v11 )
                break;
              v12 = *(_DWORD *)(v11 + 112);
              v13 = v10[2];
              if ( v12 >= 0 )
                goto LABEL_14;
LABEL_21:
              v14 = *(_QWORD *)(*(_QWORD *)(v13 + 24) + 16LL * (v12 & 0x7FFFFFFF) + 8);
              if ( !v14 )
                goto LABEL_22;
LABEL_15:
              if ( (*(_BYTE *)(v14 + 4) & 8) != 0 )
              {
                while ( 1 )
                {
                  v14 = *(_QWORD *)(v14 + 32);
                  if ( !v14 )
                    break;
                  if ( (*(_BYTE *)(v14 + 4) & 8) == 0 )
                    goto LABEL_16;
                }
                v23 = *(void (**)())(*v10 + 64LL);
                if ( v23 == nullsub_737 )
                  goto LABEL_23;
LABEL_40:
                ((void (__fastcall *)(_QWORD *, __int64))v23)(v10, v11);
                v12 = *(_DWORD *)(v11 + 112);
                goto LABEL_23;
              }
LABEL_16:
              ++v9;
              (*(void (__fastcall **)(_QWORD *, __int64))(*v10 + 40LL))(v10, v11);
              if ( v56 == v9 )
                goto LABEL_34;
            }
            v21 = v18 + 1;
            if ( (unsigned int)v17 >= v18 + 1 )
              goto LABEL_19;
            if ( v21 < v17 )
              break;
            if ( v21 <= v17 )
              goto LABEL_19;
            if ( v21 > (unsigned __int64)*(unsigned int *)(v16 + 412) )
            {
              v50 = v18;
              v52 = 8LL * v18;
              v55 = *v9;
              sub_16CD150(v16 + 400, (const void *)(v16 + 416), v21, 8, v8, v18);
              v17 = *(unsigned int *)(v16 + 408);
              v19 = v50;
              v20 = v52;
              v15 = v55;
            }
            v22 = *(_QWORD *)(v16 + 400);
            v39 = (_QWORD *)(v22 + 8 * v17);
            v40 = (_QWORD *)(v22 + 8LL * v21);
            v41 = *(_QWORD *)(v16 + 416);
            if ( v40 != v39 )
            {
              do
                *v39++ = v41;
              while ( v40 != v39 );
              v22 = *(_QWORD *)(v16 + 400);
            }
            *(_DWORD *)(v16 + 408) = v21;
LABEL_20:
            v53 = v19;
            *(_QWORD *)(v22 + v20) = sub_1DBA290(v15);
            v11 = *(_QWORD *)(*(_QWORD *)(v16 + 400) + 8 * v53);
            sub_1DBB110((_QWORD *)v16, v11);
            v12 = *(_DWORD *)(v11 + 112);
            v13 = v10[2];
            if ( v12 < 0 )
              goto LABEL_21;
LABEL_14:
            v14 = *(_QWORD *)(*(_QWORD *)(v13 + 272) + 8LL * (unsigned int)v12);
            if ( v14 )
              goto LABEL_15;
LABEL_22:
            v23 = *(void (**)())(*v10 + 64LL);
            if ( v23 != nullsub_737 )
              goto LABEL_40;
LABEL_23:
            v24 = v10[4];
            v54 = 8LL * (v12 & 0x7FFFFFFF);
            v25 = (__int64 *)(*(_QWORD *)(v24 + 400) + v54);
            v26 = (unsigned __int64 *)*v25;
            if ( *v25 )
            {
              sub_1DB4CE0(*v25);
              v27 = v26[12];
              v51 = v27;
              if ( v27 )
              {
                v28 = *(_QWORD *)(v27 + 16);
                while ( v28 )
                {
                  sub_210BA00(*(_QWORD *)(v28 + 24));
                  v29 = v28;
                  v28 = *(_QWORD *)(v28 + 16);
                  j_j___libc_free_0(v29, 56);
                }
                j_j___libc_free_0(v51, 48);
              }
              v30 = v26[8];
              if ( (unsigned __int64 *)v30 != v26 + 10 )
                _libc_free(v30);
              if ( (unsigned __int64 *)*v26 != v26 + 2 )
                _libc_free(*v26);
              j_j___libc_free_0(v26, 120);
              v25 = (__int64 *)(*(_QWORD *)(v24 + 400) + v54);
            }
            *v25 = 0;
            if ( v56 == ++v9 )
            {
LABEL_34:
              v1 = v10;
              v56 = v57;
              goto LABEL_35;
            }
          }
          *(_DWORD *)(v16 + 408) = v21;
LABEL_19:
          v22 = *(_QWORD *)(v16 + 400);
          goto LABEL_20;
        }
LABEL_35:
        v31 = (unsigned __int64)v56;
        if ( v56 != (int *)v59 )
LABEL_36:
          _libc_free(v31);
      }
    }
  }
}
