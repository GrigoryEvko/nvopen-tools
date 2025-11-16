// Function: sub_2F636C0
// Address: 0x2f636c0
//
__int64 __fastcall sub_2F636C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 result; // rax
  __int64 *v9; // r12
  __int64 v10; // r15
  __int64 v11; // r14
  unsigned int *v12; // rdx
  __int64 v13; // r10
  _QWORD *v14; // rdx
  __int64 v15; // r13
  _QWORD *v16; // rax
  __int64 *v17; // rax
  __int64 v18; // r10
  __int64 v19; // rsi
  __int64 v20; // r14
  __int64 v21; // r13
  _QWORD *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r15
  __int64 *v26; // rdi
  __int64 *v27; // rdx
  unsigned int v28; // eax
  __int64 *v29; // rsi
  __int64 v30; // r10
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  int v33; // r14d
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 **v37; // [rsp+10h] [rbp-90h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+18h] [rbp-88h]
  const void *v40; // [rsp+20h] [rbp-80h]
  unsigned __int64 v41; // [rsp+28h] [rbp-78h]
  __int64 v42; // [rsp+28h] [rbp-78h]
  __int64 v44; // [rsp+48h] [rbp-58h]
  _QWORD *v45; // [rsp+48h] [rbp-58h]
  unsigned int v46; // [rsp+48h] [rbp-58h]
  _QWORD *v47; // [rsp+48h] [rbp-58h]
  unsigned __int64 v48; // [rsp+48h] [rbp-58h]
  __int64 *v49; // [rsp+50h] [rbp-50h]
  __int64 v50; // [rsp+58h] [rbp-48h]
  __int64 v51; // [rsp+60h] [rbp-40h] BYREF
  __int64 v52; // [rsp+68h] [rbp-38h] BYREF

  v6 = *a1;
  v49 = (__int64 *)a4;
  result = *(unsigned int *)(*a1 + 72);
  if ( (_DWORD)result )
  {
    v9 = a1;
    v10 = 0;
    v50 = 8LL * (unsigned int)(result - 1);
    v40 = (const void *)(a3 + 16);
    while ( 1 )
    {
      v11 = *(_QWORD *)(*(_QWORD *)(v6 + 64) + v10);
      v12 = (unsigned int *)(v9[16] + 8 * v10);
      v13 = *(_QWORD *)(v11 + 8);
      result = *v12;
      if ( !(_DWORD)result )
        break;
      if ( (_DWORD)result == 1 )
      {
        v14 = (_QWORD *)(v13 & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_6;
      }
LABEL_13:
      if ( v10 == v50 )
        return result;
      v6 = *v9;
      v10 += 8;
    }
    if ( !*((_BYTE *)v12 + 56) || !*((_BYTE *)v12 + 57) )
      goto LABEL_13;
    if ( !v49 )
    {
      v48 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      sub_2E0A600(v6, v11);
      *(_QWORD *)(v11 + 8) = 0;
      v14 = (_QWORD *)v48;
      goto LABEL_6;
    }
    v41 = v13 & 0xFFFFFFFFFFFFFFF8LL;
    v44 = *(_QWORD *)(v11 + 8);
    v17 = (__int64 *)sub_2E09D00((__int64 *)v6, v44);
    v18 = v44;
    v19 = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
    if ( v17 != (__int64 *)v19
      && (*(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v17 >> 1) & 3) <= (*(_DWORD *)(v41 + 24)
                                                                                             | (unsigned int)(v44 >> 1)
                                                                                             & 3) )
    {
      v19 = (__int64)v17;
    }
    v45 = (_QWORD *)v41;
    v38 = v18;
    v42 = *(_QWORD *)(v19 + 8);
    sub_2E0A600(*v9, v11);
    v14 = v45;
    *(_QWORD *)(v11 + 8) = 0;
    a4 = v49[13];
    if ( a4 )
    {
      v37 = (__int64 **)v9;
      v20 = 0;
      v21 = 0;
      v36 = a2;
      v22 = v45;
      v23 = v38;
      v39 = v10;
      v24 = v23;
      v25 = v49[13];
      v51 = 0;
      v52 = 0;
      v46 = (v23 >> 1) & 3;
      while ( 1 )
      {
        while ( 1 )
        {
          v27 = (__int64 *)sub_2E09D00((__int64 *)v25, v24);
          if ( v27 != (__int64 *)(*(_QWORD *)v25 + 24LL * *(unsigned int *)(v25 + 8)) )
            break;
LABEL_27:
          v25 = *(_QWORD *)(v25 + 104);
          if ( !v25 )
            goto LABEL_34;
        }
        v26 = v27;
        v28 = *(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v27 >> 1) & 3;
        if ( v28 > (*((_DWORD *)v22 + 6) | v46) )
        {
          if ( (v21 & 0xFFFFFFFFFFFFFFF8LL) != 0
            && v28 >= (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3) )
          {
            v26 = &v51;
          }
          v21 = *v26;
          v51 = *v26;
          goto LABEL_27;
        }
        v29 = v27 + 1;
        if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0
          && (*(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v20 >> 1) & 3) >= (*(_DWORD *)((v27[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v27[1] >> 1)
                                                                                               & 3) )
        {
          v29 = &v52;
        }
        v20 = *v29;
        v25 = *(_QWORD *)(v25 + 104);
        v52 = *v29;
        if ( !v25 )
        {
LABEL_34:
          v14 = v22;
          v30 = v24;
          v10 = v39;
          v9 = (__int64 *)v37;
          a2 = v36;
          v31 = v20 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            a5 = v42;
            a4 = *(_DWORD *)(v31 + 24) | (unsigned int)(v20 >> 1) & 3;
            if ( (unsigned int)a4 >= (*(_DWORD *)((v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v42 >> 1) & 3) )
              v20 = v42;
            v42 = v20;
          }
          if ( (v21 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            a6 = v42;
            a4 = *(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3;
            if ( (unsigned int)a4 >= (*(_DWORD *)((v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v42 >> 1) & 3) )
              v21 = v42;
            v42 = v21;
          }
          if ( v31 )
          {
            v47 = v14;
            v32 = sub_2E09D00(*v37, v30);
            a4 = (__int64)*v37;
            v14 = v47;
            if ( v32 != **v37 )
            {
              a4 = v42;
              *(_QWORD *)(v32 - 16) = v42;
            }
          }
          break;
        }
      }
    }
LABEL_6:
    v15 = v14[2];
    if ( *(_WORD *)(v15 + 68) == 20 )
    {
      v33 = *(_DWORD *)(*(_QWORD *)(v15 + 32) + 48LL);
      if ( v33 < 0 )
      {
        v34 = v9[6];
        if ( *(_DWORD *)(v34 + 12) != v33 && *(_DWORD *)(v34 + 8) != v33 )
        {
          v35 = *(unsigned int *)(a3 + 8);
          if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, v40, v35 + 1, 4u, a5, a6);
            v35 = *(unsigned int *)(a3 + 8);
          }
          a4 = a3;
          v14 = *(_QWORD **)a3;
          *(_DWORD *)(*(_QWORD *)a3 + 4 * v35) = v33;
          ++*(_DWORD *)(a3 + 8);
        }
      }
    }
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_45;
    v16 = *(_QWORD **)(a2 + 8);
    a4 = *(unsigned int *)(a2 + 20);
    v14 = &v16[a4];
    if ( v16 != v14 )
    {
      while ( v15 != *v16 )
      {
        if ( v14 == ++v16 )
          goto LABEL_46;
      }
      goto LABEL_12;
    }
LABEL_46:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = a4 + 1;
      *v14 = v15;
      ++*(_QWORD *)a2;
    }
    else
    {
LABEL_45:
      sub_C8CC70(a2, v15, (__int64)v14, a4, a5, a6);
    }
LABEL_12:
    sub_2FAD510(*(_QWORD *)(v9[7] + 32), v15, 0);
    result = sub_2E88E20(v15);
    goto LABEL_13;
  }
  return result;
}
