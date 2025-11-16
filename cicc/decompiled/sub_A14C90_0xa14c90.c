// Function: sub_A14C90
// Address: 0xa14c90
//
__int64 __fastcall sub_A14C90(unsigned __int64 a1, unsigned __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  unsigned int v9; // r12d
  __int64 v10; // r9
  unsigned __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  bool v16; // zf
  _QWORD *v17; // r15
  __int64 *v18; // rax
  __int64 *v19; // rcx
  __int64 *v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r9d
  __int64 v26; // rdx
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v31; // [rsp+8h] [rbp-B8h]
  __int64 v32; // [rsp+10h] [rbp-B0h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  __int64 *v34; // [rsp+18h] [rbp-A8h]
  unsigned int v35; // [rsp+18h] [rbp-A8h]
  unsigned int v36; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+28h] [rbp-98h] BYREF
  __int64 v40; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v41; // [rsp+38h] [rbp-88h]
  unsigned __int64 v42; // [rsp+40h] [rbp-80h] BYREF
  __int64 v43; // [rsp+48h] [rbp-78h] BYREF
  __int64 v44; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+58h] [rbp-68h] BYREF
  __int64 v46; // [rsp+60h] [rbp-60h] BYREF
  __int64 v47; // [rsp+68h] [rbp-58h]
  __int64 v48; // [rsp+70h] [rbp-50h]
  unsigned int v49; // [rsp+78h] [rbp-48h]
  __int16 v50; // [rsp+80h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 24) <= (unsigned int)a2 )
    return 0;
  v5 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)a1;
  v8 = a1;
  v9 = a2;
  v10 = a4;
  v11 = (v5 - *(_QWORD *)a1) >> 5;
  if ( (unsigned int)a2 >= (unsigned int)v11 )
  {
    a2 = (unsigned int)(a2 + 1);
    if ( a2 > v11 )
    {
      a2 -= v11;
      v33 = a5;
      sub_9C9970((__int64 *)a1, a2);
      v7 = *(_QWORD *)a1;
      v10 = a4;
      a5 = v33;
    }
    else if ( a2 < v11 )
    {
      a2 *= 32LL;
      v31 = v7 + a2;
      if ( v5 != v7 + a2 )
      {
        v22 = v7 + a2;
        do
        {
          v23 = *(_QWORD *)(v22 + 16);
          LOBYTE(a2) = v23 != -4096;
          if ( ((v23 != 0) & (unsigned __int8)a2) != 0 && v23 != -8192 )
          {
            a1 = v22;
            v32 = a5;
            v35 = v10;
            sub_BD60C0(v22);
            a5 = v32;
            v10 = v35;
          }
          v22 += 32LL;
        }
        while ( v5 != v22 );
        v7 = *(_QWORD *)v8;
        *(_QWORD *)(v8 + 8) = v31;
      }
    }
  }
  v12 = 32LL * v9;
  v13 = *(_QWORD *)(v7 + v12 + 16);
  if ( !v13 )
  {
    v36 = v10;
    if ( !a3 )
      return 0;
    v50 = 257;
    v24 = sub_22077B0(40);
    v14 = v24;
    if ( v24 )
    {
      sub_B2BA90(v24, a3, &v46, 0, 0, v36);
      v48 = v14;
      v25 = v36;
      v46 = 6;
      v47 = 0;
      if ( v14 == -4096 || v14 == -8192 )
      {
        v26 = v14;
      }
      else
      {
        sub_BD73F0(&v46);
        v26 = v48;
        v25 = v36;
      }
      v27 = *(_QWORD *)v8 + v12;
      v49 = v25;
      v28 = *(_QWORD *)(v27 + 16);
      if ( v26 == v28 )
      {
LABEL_47:
        *(_DWORD *)(v27 + 24) = v25;
        if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
          sub_BD60C0(&v46);
        return v14;
      }
      if ( v28 == -4096 || v28 == 0 || v28 == -8192 )
        goto LABEL_43;
    }
    else
    {
      v27 = *(_QWORD *)v8 + v12;
      v46 = 6;
      v47 = 0;
      v48 = 0;
      v49 = v36;
      v29 = *(_QWORD *)(v27 + 16);
      if ( !v29 )
      {
        *(_DWORD *)(v27 + 24) = v36;
        return v14;
      }
      if ( v29 == -4096 || v29 == -8192 )
      {
        *(_QWORD *)(v27 + 16) = 0;
LABEL_46:
        v25 = v49;
        v28 = v48;
        goto LABEL_47;
      }
    }
    sub_BD60C0(v27);
    v26 = v48;
LABEL_43:
    *(_QWORD *)(v27 + 16) = v26;
    if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
      sub_BD6050(v27, v46 & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_46;
  }
  if ( !a3 || (v14 = 0, a3 == *(_QWORD *)(v13 + 8)) )
  {
    v16 = *(_QWORD *)(v8 + 48) == 0;
    LODWORD(v44) = v9;
    v45 = a5;
    if ( v16 )
      sub_4263D6(a1, a2, v7);
    (*(void (__fastcall **)(__int64 *, unsigned __int64, __int64 *, __int64 *, __int64, __int64))(v8 + 56))(
      &v46,
      v8 + 32,
      &v44,
      &v45,
      a5,
      v10);
    v14 = v46;
    if ( (v47 & 1) != 0 )
    {
      v46 = 0;
      LOBYTE(v47) = v47 & 0xFD;
      v38 = 0;
      v39 = 0;
      v17 = (_QWORD *)(v14 & 0xFFFFFFFFFFFFFFFELL);
      if ( v17 )
      {
        v40 = 0;
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, void *))(*v17 + 48LL))(v17, &unk_4F84052) )
        {
          v18 = (__int64 *)v17[2];
          v19 = (__int64 *)v17[1];
          v41 = 1;
          v30 = v18;
          if ( v19 == v18 )
          {
            v21 = 1;
          }
          else
          {
            do
            {
              v34 = v19;
              v44 = *v19;
              *v19 = 0;
              sub_A14970(&v43, &v44);
              v45 = v41 | 1;
              sub_9CDB40(&v42, (unsigned __int64 *)&v45, (unsigned __int64 *)&v43);
              v20 = v34;
              v41 = v42 | 1;
              if ( (v45 & 1) != 0 || (v45 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v45);
              if ( (v43 & 1) != 0 || (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                sub_C63C30(&v43);
              if ( v44 )
              {
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v44 + 8LL))(v44);
                v20 = v34;
              }
              v19 = v20 + 1;
            }
            while ( v30 != v19 );
            v21 = v41 | 1;
          }
          v44 = v21;
          (*(void (__fastcall **)(_QWORD *))(*v17 + 8LL))(v17);
        }
        else
        {
          v45 = (__int64)v17;
          sub_A14970(&v44, &v45);
          if ( v45 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v45 + 8LL))(v45);
        }
        if ( (v44 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          BUG();
        if ( (v40 & 1) != 0 || (v40 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v40);
      }
      if ( (v39 & 1) != 0 || (v39 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v39);
      if ( (v38 & 1) != 0 || (v38 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v38);
      if ( (v47 & 2) != 0 )
        sub_9D21E0(&v46);
      if ( (v47 & 1) != 0 )
      {
        v14 = v46;
        if ( v46 )
        {
          v14 = 0;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v46 + 8LL))(v46);
        }
        return v14;
      }
      return 0;
    }
  }
  return v14;
}
