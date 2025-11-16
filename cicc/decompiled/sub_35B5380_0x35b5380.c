// Function: sub_35B5380
// Address: 0x35b5380
//
__int64 __fastcall sub_35B5380(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  int v4; // edx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // r8
  __int64 v9; // r9
  int *v10; // rdi
  int *v11; // rbx
  __int64 v12; // r14
  int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rax
  int v16; // r10d
  __int64 v17; // r12
  unsigned __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 v20; // r13
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rax
  void (*v24)(); // rax
  __int64 v25; // rdx
  unsigned __int64 *v26; // r14
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  void (*v31)(); // rax
  __int64 *v32; // rbx
  unsigned __int64 *v33; // r12
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdx
  __int64 v37; // r14
  unsigned __int64 v38; // rax
  _QWORD *v39; // rdx
  _QWORD *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rcx
  __int64 v44; // rax
  __int64 v45; // r9
  unsigned __int16 v46; // ax
  unsigned __int64 v47; // [rsp+0h] [rbp-70h]
  _QWORD *v48; // [rsp+10h] [rbp-60h]
  int v49; // [rsp+10h] [rbp-60h]
  int *v50; // [rsp+18h] [rbp-58h]
  int *v51; // [rsp+20h] [rbp-50h] BYREF
  __int64 v52; // [rsp+28h] [rbp-48h]
  _BYTE v53[64]; // [rsp+30h] [rbp-40h] BYREF

  sub_35B4F70((__int64)a1);
  while ( 1 )
  {
    result = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 48LL))(a1);
    v3 = result;
    if ( !result )
      return result;
    v4 = *(_DWORD *)(result + 112);
    v5 = a1[2];
    if ( v4 < 0 )
      v6 = *(_QWORD *)(*(_QWORD *)(v5 + 56) + 16LL * (v4 & 0x7FFFFFFF) + 8);
    else
      v6 = *(_QWORD *)(*(_QWORD *)(v5 + 304) + 8LL * (unsigned int)v4);
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
      v31 = *(void (**)())(*a1 + 64LL);
      if ( v31 != nullsub_1898 )
      {
        ((void (__fastcall *)(_QWORD *, __int64))v31)(a1, v3);
        v4 = *(_DWORD *)(v3 + 112);
      }
      v32 = (__int64 *)(*(_QWORD *)(a1[4] + 152LL) + 8LL * (v4 & 0x7FFFFFFF));
      v33 = (unsigned __int64 *)*v32;
      if ( *v32 )
      {
        sub_2E0AFD0(*v32);
        v34 = v33[12];
        if ( v34 )
        {
          sub_35B4780(*(_QWORD *)(v34 + 16));
          j_j___libc_free_0(v34);
        }
        v35 = v33[8];
        if ( (unsigned __int64 *)v35 != v33 + 10 )
          _libc_free(v35);
        if ( (unsigned __int64 *)*v33 != v33 + 2 )
          _libc_free(*v33);
        j_j___libc_free_0((unsigned __int64)v33);
      }
      *v32 = 0;
    }
    else
    {
LABEL_7:
      ++*(_DWORD *)(a1[5] + 24LL);
      v51 = (int *)v53;
      v52 = 0x400000000LL;
      v7 = (*(__int64 (__fastcall **)(_QWORD *, __int64, int **))(*a1 + 56LL))(a1, v3, &v51);
      if ( v7 == -1 )
      {
        v41 = *(unsigned int *)(v3 + 112);
        v42 = a1[2];
        v43 = (_QWORD *)(*(_QWORD *)(v42 + 56) + 16LL * (*(_DWORD *)(v3 + 112) & 0x7FFFFFFF));
        if ( (int)v41 < 0 )
          v44 = v43[1];
        else
          v44 = *(_QWORD *)(*(_QWORD *)(v42 + 304) + 8 * v41);
        if ( v44 )
        {
          v45 = *(_QWORD *)(v44 + 16);
LABEL_69:
          if ( (unsigned int)*(unsigned __int16 *)(v45 + 68) - 1 > 1 )
          {
            while ( 1 )
            {
              v44 = *(_QWORD *)(v44 + 32);
              if ( !v44 )
                break;
              if ( *(_QWORD *)(v44 + 16) != v45 )
              {
                v45 = *(_QWORD *)(v44 + 16);
                goto LABEL_69;
              }
            }
          }
        }
        else
        {
          v45 = 0;
        }
        v46 = sub_35B51A0((__int64)a1, (unsigned __int16 ***)(*v43 & 0xFFFFFFFFFFFFFFF8LL), v45);
        sub_35B4BD0((__int64)a1, *(_DWORD *)(v3 + 112), v46);
      }
      else if ( v7 )
      {
        sub_2E20EE0((_QWORD *)a1[5], v3, v7);
      }
      v10 = v51;
      v50 = &v51[(unsigned int)v52];
      if ( v50 != v51 )
      {
        v11 = v51;
        while ( 1 )
        {
          while ( 1 )
          {
            v16 = *v11;
            v17 = a1[4];
            v18 = *(unsigned int *)(v17 + 160);
            v19 = *v11 & 0x7FFFFFFF;
            v20 = 8LL * v19;
            if ( v19 >= (unsigned int)v18 )
              break;
            v12 = *(_QWORD *)(*(_QWORD *)(v17 + 152) + 8LL * v19);
            if ( !v12 )
              break;
            v13 = *(_DWORD *)(v12 + 112);
            v14 = a1[2];
            if ( v13 >= 0 )
              goto LABEL_14;
LABEL_21:
            v15 = *(_QWORD *)(*(_QWORD *)(v14 + 56) + 16LL * (v13 & 0x7FFFFFFF) + 8);
            if ( !v15 )
              goto LABEL_22;
LABEL_15:
            if ( (*(_BYTE *)(v15 + 4) & 8) != 0 )
            {
              while ( 1 )
              {
                v15 = *(_QWORD *)(v15 + 32);
                if ( !v15 )
                  break;
                if ( (*(_BYTE *)(v15 + 4) & 8) == 0 )
                  goto LABEL_16;
              }
              v24 = *(void (**)())(*a1 + 64LL);
              if ( v24 == nullsub_1898 )
                goto LABEL_23;
LABEL_40:
              ((void (__fastcall *)(_QWORD *, __int64))v24)(a1, v12);
              v13 = *(_DWORD *)(v12 + 112);
              goto LABEL_23;
            }
LABEL_16:
            ++v11;
            sub_35B4EE0((__int64)a1, v12);
            if ( v50 == v11 )
              goto LABEL_34;
          }
          v21 = v19 + 1;
          if ( (unsigned int)v18 >= v21 )
            goto LABEL_19;
          v36 = v21;
          if ( v21 == v18 )
            goto LABEL_19;
          if ( v21 < v18 )
            break;
          v37 = *(_QWORD *)(v17 + 168);
          v38 = v21 - v18;
          if ( v36 > *(unsigned int *)(v17 + 164) )
          {
            v47 = v38;
            v49 = *v11;
            sub_C8D5F0(v17 + 152, (const void *)(v17 + 168), v36, 8u, v8, v9);
            v38 = v47;
            v16 = v49;
            v18 = *(unsigned int *)(v17 + 160);
          }
          v22 = *(_QWORD *)(v17 + 152);
          v39 = (_QWORD *)(v22 + 8 * v18);
          v40 = &v39[v38];
          if ( v39 != v40 )
          {
            do
              *v39++ = v37;
            while ( v40 != v39 );
            LODWORD(v18) = *(_DWORD *)(v17 + 160);
            v22 = *(_QWORD *)(v17 + 152);
          }
          *(_DWORD *)(v17 + 160) = v38 + v18;
LABEL_20:
          v23 = sub_2E10F30(v16);
          *(_QWORD *)(v22 + v20) = v23;
          v12 = v23;
          sub_2E11E80((_QWORD *)v17, v23);
          v13 = *(_DWORD *)(v12 + 112);
          v14 = a1[2];
          if ( v13 < 0 )
            goto LABEL_21;
LABEL_14:
          v15 = *(_QWORD *)(*(_QWORD *)(v14 + 304) + 8LL * (unsigned int)v13);
          if ( v15 )
            goto LABEL_15;
LABEL_22:
          v24 = *(void (**)())(*a1 + 64LL);
          if ( v24 != nullsub_1898 )
            goto LABEL_40;
LABEL_23:
          v25 = v13 & 0x7FFFFFFF;
          v26 = *(unsigned __int64 **)(*(_QWORD *)(a1[4] + 152LL) + 8 * v25);
          v48 = (_QWORD *)(*(_QWORD *)(a1[4] + 152LL) + 8 * v25);
          if ( v26 )
          {
            sub_2E0AFD0((__int64)v26);
            v27 = v26[12];
            if ( v27 )
            {
              v28 = *(_QWORD *)(v27 + 16);
              while ( v28 )
              {
                sub_35B4780(*(_QWORD *)(v28 + 24));
                v29 = v28;
                v28 = *(_QWORD *)(v28 + 16);
                j_j___libc_free_0(v29);
              }
              j_j___libc_free_0(v27);
            }
            v30 = v26[8];
            if ( (unsigned __int64 *)v30 != v26 + 10 )
              _libc_free(v30);
            if ( (unsigned __int64 *)*v26 != v26 + 2 )
              _libc_free(*v26);
            j_j___libc_free_0((unsigned __int64)v26);
          }
          ++v11;
          *v48 = 0;
          if ( v50 == v11 )
          {
LABEL_34:
            v10 = v51;
            goto LABEL_35;
          }
        }
        *(_DWORD *)(v17 + 160) = v21;
LABEL_19:
        v22 = *(_QWORD *)(v17 + 152);
        goto LABEL_20;
      }
LABEL_35:
      if ( v10 != (int *)v53 )
        _libc_free((unsigned __int64)v10);
    }
  }
}
