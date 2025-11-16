// Function: sub_39DA7B0
// Address: 0x39da7b0
//
__int64 __fastcall sub_39DA7B0(__int64 a1, unsigned __int64 *a2)
{
  int v3; // ebx
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int64 *v8; // r9
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // r12
  char *v12; // rdx
  __int64 v13; // rax
  void (__fastcall *v14)(__int64, char **, _QWORD); // r12
  unsigned int v15; // eax
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r9
  char *v21; // rdi
  char *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 *v27; // rax
  size_t v28; // rdx
  unsigned __int64 v29; // [rsp+8h] [rbp-C8h]
  __int64 v30; // [rsp+10h] [rbp-C0h]
  unsigned __int64 *v31; // [rsp+10h] [rbp-C0h]
  void *src; // [rsp+18h] [rbp-B8h]
  void *srca; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *srcb; // [rsp+18h] [rbp-B8h]
  __int64 v36; // [rsp+28h] [rbp-A8h]
  __int64 v37; // [rsp+38h] [rbp-98h] BYREF
  char *v38; // [rsp+40h] [rbp-90h] BYREF
  char *v39; // [rsp+48h] [rbp-88h]
  _BYTE *v40; // [rsp+50h] [rbp-80h] BYREF
  __int64 v41; // [rsp+58h] [rbp-78h]
  _BYTE v42[16]; // [rsp+60h] [rbp-70h] BYREF
  char *v43; // [rsp+70h] [rbp-60h] BYREF
  size_t n; // [rsp+78h] [rbp-58h]
  __int64 v45; // [rsp+80h] [rbp-50h] BYREF
  __int64 v46; // [rsp+88h] [rbp-48h]
  int v47; // [rsp+90h] [rbp-40h]
  char **v48; // [rsp+98h] [rbp-38h]

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -1431655765 * ((__int64)(a2[1] - *a2) >> 4);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = 1;
    v36 = v4 + 2;
    while ( 1 )
    {
      while ( 1 )
      {
        v7 = v5 + 48;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 72LL))(
               a1,
               (unsigned int)(v6 - 1),
               &v37) )
        {
          break;
        }
        v5 += 48;
        if ( v36 == ++v6 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
      }
      v8 = (unsigned __int64 *)a2[1];
      v9 = *a2;
      v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v8 - *a2) >> 4);
      if ( v10 <= v6 - 1 )
      {
        if ( v10 < v6 )
        {
          sub_39DA530(a2, v6 - v10);
          v9 = *a2;
        }
        else if ( v10 > v6 )
        {
          v27 = (unsigned __int64 *)(v9 + v7);
          v29 = v9 + v7;
          if ( v8 != (unsigned __int64 *)(v9 + v7) )
          {
            do
            {
              if ( (unsigned __int64 *)*v27 != v27 + 2 )
              {
                v31 = v8;
                srcb = v27;
                j_j___libc_free_0(*v27);
                v8 = v31;
                v27 = srcb;
              }
              v27 += 6;
            }
            while ( v8 != v27 );
            v9 = *a2;
            a2[1] = v29;
          }
        }
      }
      v11 = v9 + v5;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
        break;
      v42[0] = 0;
      v40 = v42;
      v41 = 0;
      v43 = (char *)&unk_49EFBE0;
      v47 = 1;
      v46 = 0;
      v45 = 0;
      n = 0;
      v48 = &v40;
      sub_16E4080(a1);
      sub_16E7EE0((__int64)&v43, *(char **)v11, *(_QWORD *)(v11 + 8));
      if ( v46 != n )
        sub_16E7BA0((__int64 *)&v43);
      v12 = *v48;
      v39 = v48[1];
      v13 = *(_QWORD *)a1;
      v38 = v12;
      v14 = *(void (__fastcall **)(__int64, char **, _QWORD))(v13 + 216);
      v15 = sub_15C8A80(v12, (unsigned __int64)v39);
      v14(a1, &v38, v15);
      sub_16E7BC0((__int64 *)&v43);
      if ( v40 != v42 )
        j_j___libc_free_0((unsigned __int64)v40);
LABEL_13:
      v5 = v7;
      ++v6;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 80LL))(a1, v37);
      if ( v36 == v6 )
        return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
    }
    v17 = *(_QWORD *)a1;
    v40 = 0;
    v41 = 0;
    src = *(void **)(v17 + 216);
    v18 = sub_15C8A80(0, 0);
    ((void (__fastcall *)(__int64, _BYTE **, _QWORD))src)(a1, &v40, v18);
    v19 = sub_16E4080(a1);
    v20 = v19;
    if ( v40 )
    {
      v30 = v19;
      v43 = (char *)&v45;
      sub_39CF540((__int64 *)&v43, v40, (__int64)&v40[v41]);
      v21 = *(char **)v11;
      v20 = v30;
      v22 = *(char **)v11;
      if ( v43 != (char *)&v45 )
      {
        if ( v21 == (char *)(v11 + 16) )
        {
          *(_QWORD *)v11 = v43;
          *(_QWORD *)(v11 + 8) = n;
          *(_QWORD *)(v11 + 16) = v45;
        }
        else
        {
          *(_QWORD *)v11 = v43;
          v23 = *(_QWORD *)(v11 + 16);
          *(_QWORD *)(v11 + 8) = n;
          *(_QWORD *)(v11 + 16) = v45;
          if ( v21 )
          {
            v43 = v21;
            v45 = v23;
LABEL_20:
            n = 0;
            *v22 = 0;
            if ( v43 != (char *)&v45 )
            {
              srca = (void *)v20;
              j_j___libc_free_0((unsigned __int64)v43);
              v20 = (__int64)srca;
            }
            v24 = sub_16E4250(v20);
            if ( v24 )
            {
              v25 = *(_QWORD *)(v24 + 16);
              v26 = *(_QWORD *)(v24 + 24);
              *(_QWORD *)(v11 + 32) = v25;
              *(_QWORD *)(v11 + 40) = v26;
            }
            goto LABEL_13;
          }
        }
        v43 = (char *)&v45;
        v22 = (char *)&v45;
        goto LABEL_20;
      }
      v28 = n;
      if ( n )
      {
        if ( n == 1 )
        {
          *v21 = v45;
          v28 = n;
          v21 = *(char **)v11;
        }
        else
        {
          memcpy(v21, &v45, n);
          v28 = n;
          v21 = *(char **)v11;
          v20 = v30;
        }
      }
    }
    else
    {
      n = 0;
      v28 = 0;
      v43 = (char *)&v45;
      LOBYTE(v45) = 0;
      v21 = *(char **)v11;
    }
    *(_QWORD *)(v11 + 8) = v28;
    v21[v28] = 0;
    v22 = v43;
    goto LABEL_20;
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
}
