// Function: sub_96E6C0
// Address: 0x96e6c0
//
__int64 __fastcall sub_96E6C0(unsigned int a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  unsigned __int64 v8; // rdi
  unsigned int v9; // ebx
  __int64 v10; // r14
  unsigned __int64 v11; // r14
  __int64 v12; // r14
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdi
  bool v20; // bl
  unsigned int v21; // ebx
  __int64 v22; // r14
  __int64 v23; // r14
  bool v24; // bl
  unsigned int v25; // ebx
  int v26; // r14d
  int v27; // eax
  unsigned int v28; // [rsp+8h] [rbp-C8h]
  __int64 v29; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+28h] [rbp-A8h] BYREF
  unsigned __int64 v31; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-98h]
  __int64 v33; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v34; // [rsp+48h] [rbp-88h]
  __int64 v35; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+58h] [rbp-78h]
  unsigned __int64 v37; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+68h] [rbp-68h]
  __int64 v39; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-58h]
  __int64 v41; // [rsp+80h] [rbp-50h] BYREF
  __int64 v42; // [rsp+88h] [rbp-48h]
  __int64 v43; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+98h] [rbp-38h]

  if ( *(_BYTE *)a2 != 5 && *a3 != 5 )
    goto LABEL_3;
  if ( a1 == 28 )
  {
    sub_9AC3E0((unsigned int)&v37, a2, a4, 0, 0, 0, 0, 1);
    sub_9AC3E0((unsigned int)&v41, (_DWORD)a3, a4, 0, 0, 0, 0, 1);
    v9 = v44;
    v34 = v44;
    if ( v44 > 0x40 )
    {
      sub_C43780(&v33, &v43);
      v9 = v34;
      if ( v34 > 0x40 )
      {
        sub_C43BD0(&v33, &v37);
        v9 = v34;
        v11 = v33;
        v34 = 0;
        v36 = v9;
        v35 = v33;
        if ( !v9 )
          goto LABEL_20;
        if ( v9 > 0x40 )
        {
          v20 = v9 == (unsigned int)sub_C445E0(&v35);
          if ( v11 )
          {
            j_j___libc_free_0_0(v11);
            if ( v34 > 0x40 )
            {
              if ( v33 )
                j_j___libc_free_0_0(v33);
            }
          }
LABEL_55:
          if ( v20 )
            goto LABEL_20;
          v21 = v40;
          v34 = v40;
          if ( v40 > 0x40 )
          {
            sub_C43780(&v33, &v39);
            v21 = v34;
            if ( v34 > 0x40 )
            {
              sub_C43BD0(&v33, &v41);
              v21 = v34;
              v23 = v33;
              v34 = 0;
              v36 = v21;
              v35 = v33;
              if ( !v21 )
                goto LABEL_59;
              if ( v21 > 0x40 )
              {
                v24 = v21 == (unsigned int)sub_C445E0(&v35);
                if ( v23 )
                {
                  j_j___libc_free_0_0(v23);
                  if ( v34 > 0x40 )
                  {
                    if ( v33 )
                      j_j___libc_free_0_0(v33);
                  }
                }
LABEL_74:
                if ( !v24 )
                {
                  sub_C7BCF0(&v37, &v41);
                  v25 = v38;
                  if ( v38 > 0x40 )
                    v26 = sub_C44630(&v37);
                  else
                    v26 = sub_39FAC40(v37);
                  if ( v40 > 0x40 )
                    v27 = sub_C44630(&v39);
                  else
                    v27 = sub_39FAC40(v39);
                  if ( v26 + v27 != v25 )
                  {
                    if ( v44 > 0x40 && v43 )
                      j_j___libc_free_0_0(v43);
                    if ( (unsigned int)v42 > 0x40 && v41 )
                      j_j___libc_free_0_0(v41);
                    if ( v40 > 0x40 && v39 )
                      j_j___libc_free_0_0(v39);
                    if ( v38 <= 0x40 )
                      goto LABEL_3;
                    v8 = v37;
                    if ( !v37 )
                      goto LABEL_3;
LABEL_16:
                    j_j___libc_free_0_0(v8);
                    goto LABEL_3;
                  }
                  v12 = sub_AD8D80(*(_QWORD *)(a2 + 8), &v39);
LABEL_21:
                  if ( v44 > 0x40 && v43 )
                    j_j___libc_free_0_0(v43);
                  if ( (unsigned int)v42 > 0x40 && v41 )
                    j_j___libc_free_0_0(v41);
                  if ( v40 > 0x40 && v39 )
                    j_j___libc_free_0_0(v39);
                  if ( v38 <= 0x40 )
                    goto LABEL_33;
                  v13 = v37;
                  if ( !v37 )
                    goto LABEL_33;
LABEL_32:
                  j_j___libc_free_0_0(v13);
                  goto LABEL_33;
                }
LABEL_59:
                v12 = (__int64)a3;
                goto LABEL_21;
              }
LABEL_96:
              v24 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21) == v23;
              goto LABEL_74;
            }
            v22 = v33;
          }
          else
          {
            v22 = v39;
          }
          v23 = v41 | v22;
          v36 = v21;
          v33 = v23;
          v35 = v23;
          v34 = 0;
          if ( !v21 )
            goto LABEL_59;
          goto LABEL_96;
        }
LABEL_54:
        v20 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == v11;
        goto LABEL_55;
      }
      v10 = v33;
    }
    else
    {
      v10 = v43;
    }
    v11 = v37 | v10;
    v36 = v9;
    v33 = v11;
    v35 = v11;
    v34 = 0;
    if ( !v9 )
    {
LABEL_20:
      v12 = a2;
      goto LABEL_21;
    }
    goto LABEL_54;
  }
  if ( a1 != 15 )
    goto LABEL_3;
  v32 = 1;
  v31 = 0;
  v34 = 1;
  v33 = 0;
  if ( !(unsigned __int8)sub_96E080(a2, &v29, (__int64)&v31, a4, 0)
    || !(unsigned __int8)sub_96E080((__int64)a3, &v30, (__int64)&v33, a4, 0)
    || v29 != v30 )
  {
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
    if ( v32 <= 0x40 )
      goto LABEL_3;
    v8 = v31;
    if ( !v31 )
      goto LABEL_3;
    goto LABEL_16;
  }
  v41 = sub_9208B0(a4, *(_QWORD *)(a2 + 8));
  v42 = v14;
  v28 = sub_CA1930(&v41);
  sub_C44AB0(&v37, &v33, v28);
  sub_C44AB0(&v35, &v31, v28);
  if ( v38 > 0x40 )
  {
    sub_C43D10(&v37, &v31, v38, v15, v16);
  }
  else
  {
    v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v38) & ~v37;
    if ( !v38 )
      v17 = 0;
    v37 = v17;
  }
  sub_C46250(&v37);
  sub_C45EE0(&v37, &v35);
  v18 = v38;
  v19 = *(_QWORD *)(a2 + 8);
  v38 = 0;
  LODWORD(v42) = v18;
  v41 = v37;
  v12 = sub_AD8D80(v19, &v41);
  if ( (unsigned int)v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 )
  {
    v13 = v31;
    if ( v31 )
      goto LABEL_32;
  }
LABEL_33:
  if ( v12 )
    return v12;
LABEL_3:
  if ( (unsigned __int8)sub_AC47B0(a1) )
    return sub_AD5570(a1, a2, a3, 0, 0);
  else
    return sub_AABE40(a1, a2, a3);
}
