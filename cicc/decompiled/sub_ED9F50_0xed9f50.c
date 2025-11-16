// Function: sub_ED9F50
// Address: 0xed9f50
//
__int64 __fastcall sub_ED9F50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        char a7)
{
  __int64 v11; // rsi
  unsigned __int64 v12; // r9
  unsigned __int64 v13; // rdi
  __int64 *v14; // rsi
  char v15; // al
  __int64 v16; // r9
  __int64 *v17; // rcx
  __int64 *v18; // rax
  __int64 *v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 *v23; // rcx
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // r11
  unsigned __int64 v29; // r9
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // rax
  __int64 ***v33; // rdi
  unsigned __int64 *v35; // [rsp+0h] [rbp-E0h]
  __int64 *v36; // [rsp+10h] [rbp-D0h]
  __int64 v37; // [rsp+30h] [rbp-B0h]
  __int64 v38; // [rsp+48h] [rbp-98h] BYREF
  __int64 v39; // [rsp+50h] [rbp-90h] BYREF
  __int64 v40; // [rsp+58h] [rbp-88h] BYREF
  __int64 v41; // [rsp+60h] [rbp-80h] BYREF
  __int64 v42; // [rsp+68h] [rbp-78h] BYREF
  __int64 v43; // [rsp+70h] [rbp-70h] BYREF
  __int64 v44; // [rsp+78h] [rbp-68h] BYREF
  __int64 *v45; // [rsp+80h] [rbp-60h] BYREF
  __int64 v46; // [rsp+88h] [rbp-58h]
  __int64 v47[2]; // [rsp+90h] [rbp-50h] BYREF
  __int64 **v48; // [rsp+A0h] [rbp-40h] BYREF

  v11 = *(_QWORD *)(a2 + 136);
  v45 = 0;
  v46 = 0;
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64 **))(*(_QWORD *)v11 + 24LL))(
    &v38,
    v11,
    a3,
    a4,
    &v45);
  v12 = v38 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v38 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v47[0] = a2;
    v13 = v38 & 0xFFFFFFFFFFFFFFFELL;
    v14 = (__int64 *)&unk_4F84052;
    v47[1] = (__int64)&a7;
    v48 = &v45;
    v38 = 0;
    v39 = 0;
    v37 = v12;
    v15 = (*(__int64 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v12 + 48LL))(v13, &unk_4F84052);
    v16 = v37;
    if ( v15 )
    {
      v17 = *(__int64 **)(v37 + 16);
      v18 = *(__int64 **)(v37 + 8);
      v40 = 1;
      v36 = v17;
      if ( v18 == v17 )
      {
        v21 = 1;
      }
      else
      {
        v35 = a6;
        v19 = v18;
        do
        {
          v42 = *v19;
          *v19 = 0;
          sub_ED8190(&v43, &v42, (__int64)v47);
          v20 = v40;
          v14 = &v41;
          v40 = 0;
          v41 = v20 | 1;
          sub_9CDB40((unsigned __int64 *)&v44, (unsigned __int64 *)&v41, (unsigned __int64 *)&v43);
          if ( (v40 & 1) != 0 || (v40 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v40, (__int64)&v41);
          v40 |= v44 | 1;
          if ( (v41 & 1) != 0 || (v41 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v41, (__int64)&v41);
          if ( (v43 & 1) != 0 || (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v43, (__int64)&v41);
          if ( v42 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42);
          ++v19;
        }
        while ( v36 != v19 );
        v16 = v37;
        a6 = v35;
        v21 = v40 | 1;
      }
      v43 = v21;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
    }
    else
    {
      v44 = v37;
      v14 = &v44;
      sub_ED8190(&v43, &v44, (__int64)v47);
      if ( v44 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v44 + 8LL))(v44);
    }
    if ( (v39 & 1) != 0 || (v39 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v39, (__int64)v14);
    v22 = v43 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *(_BYTE *)(a1 + 56) |= 3u;
      *(_QWORD *)a1 = v22;
      goto LABEL_44;
    }
    v43 = 0;
    sub_9C66B0(&v43);
  }
  else
  {
    v38 = 0;
  }
  v14 = v45;
  v23 = &v45[10 * v46];
  if ( v23 == v45 )
  {
LABEL_50:
    sub_ED6550(v47, byte_3F871B3);
    v31 = 13;
  }
  else
  {
    v24 = 0;
    v25 = 0;
    do
    {
      while ( 1 )
      {
        v26 = v14[9];
        if ( v26 == a5 )
        {
          *(_BYTE *)(a1 + 56) = *(_BYTE *)(a1 + 56) & 0xFC | 2;
          sub_ED82D0(a1, v14, v25);
          goto LABEL_44;
        }
        if ( ((a5 & 0x1000000000000000LL) != 0) == ((v26 & 0x1000000000000000LL) != 0) )
          break;
        v14 += 10;
        if ( v23 == v14 )
          goto LABEL_36;
      }
      if ( a6 )
      {
        v27 = (__int64 *)*v14;
        v28 = (__int64 *)v14[1];
        if ( (__int64 *)*v14 != v28 )
        {
          v29 = 0;
          do
          {
            v30 = *v27;
            if ( *v27 != -1 )
            {
              if ( v29 >= ~v30 )
              {
                v24 = -1;
                goto LABEL_35;
              }
              v29 += v30;
            }
            ++v27;
          }
          while ( v28 != v27 );
          if ( v24 < v29 )
            v24 = v29;
        }
      }
LABEL_35:
      v14 += 10;
      v25 = 1;
    }
    while ( v23 != v14 );
LABEL_36:
    if ( !(_BYTE)v25 )
      goto LABEL_50;
    if ( a6 )
      *a6 = v24;
    sub_ED6550(v47, byte_3F871B3);
    v31 = 15;
  }
  v14 = (__int64 *)a2;
  sub_ED85B0(&v44, a2, v31, v47);
  v32 = v44;
  v33 = (__int64 ***)v47[0];
  *(_BYTE *)(a1 + 56) |= 3u;
  *(_QWORD *)a1 = v32 & 0xFFFFFFFFFFFFFFFELL;
  if ( v33 != &v48 )
  {
    v14 = (__int64 *)((char *)v48 + 1);
    j_j___libc_free_0(v33, (char *)v48 + 1);
  }
LABEL_44:
  if ( (v38 & 1) != 0 || (v38 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v38, (__int64)v14);
  return a1;
}
