// Function: sub_14BBDC0
// Address: 0x14bbdc0
//
__int64 __fastcall sub_14BBDC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // eax
  __int64 v9; // r10
  unsigned int v10; // ebx
  unsigned int v11; // r13d
  int v12; // esi
  unsigned __int64 v13; // rax
  unsigned int v14; // r12d
  int v15; // eax
  unsigned __int64 v16; // rax
  unsigned int v17; // r8d
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned int v25; // [rsp+18h] [rbp-B8h]
  unsigned int v26; // [rsp+18h] [rbp-B8h]
  unsigned int v27; // [rsp+18h] [rbp-B8h]
  unsigned int v28; // [rsp+18h] [rbp-B8h]
  unsigned int v29; // [rsp+18h] [rbp-B8h]
  unsigned int v30; // [rsp+18h] [rbp-B8h]
  char v31; // [rsp+2Eh] [rbp-A2h] BYREF
  unsigned __int8 v32; // [rsp+2Fh] [rbp-A1h] BYREF
  unsigned __int64 v33; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-98h]
  unsigned __int64 v35; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-88h]
  unsigned __int64 v37; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v38; // [rsp+58h] [rbp-78h]
  unsigned __int64 v39; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-68h]
  __int64 v41; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v42; // [rsp+78h] [rbp-58h]
  unsigned __int64 v43; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+88h] [rbp-48h]
  __int64 v45; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+98h] [rbp-38h]

  v8 = sub_16431D0(*a1);
  v9 = (__int64)a1;
  v40 = v8;
  v10 = v8;
  if ( v8 > 0x40 )
  {
    sub_16A4EF0(&v39, 0, 0);
    v42 = v10;
    sub_16A4EF0(&v41, 0, 0);
    v44 = v10;
    sub_16A4EF0(&v43, 0, 0);
    v46 = v10;
    sub_16A4EF0(&v45, 0, 0);
    v9 = (__int64)a1;
  }
  else
  {
    v39 = 0;
    v42 = v8;
    v41 = 0;
    v44 = v8;
    v43 = 0;
    v46 = v8;
    v45 = 0;
  }
  sub_14BB090(v9, (__int64)&v39, a3, 0, a4, a5, a6, 0);
  sub_14BB090(a2, (__int64)&v43, a3, 0, a4, a5, a6, 0);
  v11 = v40;
  if ( v40 > 0x40 )
  {
    v12 = sub_16A5810(&v39);
  }
  else
  {
    v12 = 64;
    if ( v39 << (64 - (unsigned __int8)v40) != -1 )
    {
      _BitScanReverse64(&v13, ~(v39 << (64 - (unsigned __int8)v40)));
      v12 = v13 ^ 0x3F;
    }
  }
  v14 = v44;
  if ( v44 > 0x40 )
  {
    v15 = sub_16A5810(&v43);
  }
  else
  {
    v15 = 64;
    if ( v43 << (64 - (unsigned __int8)v44) != -1 )
    {
      _BitScanReverse64(&v16, ~(v43 << (64 - (unsigned __int8)v44)));
      v15 = v16 ^ 0x3F;
    }
  }
  v17 = 2;
  if ( v10 > v12 + v15 )
  {
    v38 = v11;
    if ( v11 > 0x40 )
    {
      sub_16A4FD0(&v37, &v39);
      v11 = v38;
      if ( v38 > 0x40 )
      {
        sub_16A8F40(&v37);
        v11 = v38;
        v19 = v37;
        v14 = v44;
LABEL_13:
        v34 = v11;
        v33 = v19;
        v38 = v14;
        if ( v14 > 0x40 )
        {
          sub_16A4FD0(&v37, &v43);
          v14 = v38;
          if ( v38 > 0x40 )
          {
            sub_16A8F40(&v37);
            v14 = v38;
            v21 = v37;
LABEL_16:
            v36 = v14;
            v35 = v21;
            sub_16AA580(&v37, &v33, &v35, &v31);
            if ( v38 > 0x40 && v37 )
              j_j___libc_free_0_0(v37);
            v17 = 2;
            if ( v31 )
            {
              sub_16AA580(&v37, &v41, &v45, &v32);
              if ( v38 > 0x40 && v37 )
                j_j___libc_free_0_0(v37);
              v17 = v32 ^ 1;
            }
            if ( v36 > 0x40 && v35 )
            {
              v25 = v17;
              j_j___libc_free_0_0(v35);
              v17 = v25;
            }
            if ( v34 > 0x40 && v33 )
            {
              v26 = v17;
              j_j___libc_free_0_0(v33);
              v17 = v26;
            }
            goto LABEL_28;
          }
          v20 = v37;
        }
        else
        {
          v20 = v43;
        }
        v21 = ~v20 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
        goto LABEL_16;
      }
      v18 = v37;
      v14 = v44;
    }
    else
    {
      v18 = v39;
    }
    v19 = ~v18 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v11);
    goto LABEL_13;
  }
LABEL_28:
  if ( v46 > 0x40 && v45 )
  {
    v27 = v17;
    j_j___libc_free_0_0(v45);
    v17 = v27;
  }
  if ( v44 > 0x40 && v43 )
  {
    v28 = v17;
    j_j___libc_free_0_0(v43);
    v17 = v28;
  }
  if ( v42 > 0x40 && v41 )
  {
    v29 = v17;
    j_j___libc_free_0_0(v41);
    v17 = v29;
  }
  if ( v40 > 0x40 && v39 )
  {
    v30 = v17;
    j_j___libc_free_0_0(v39);
    return v30;
  }
  return v17;
}
