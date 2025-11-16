// Function: sub_14C03D0
// Address: 0x14c03d0
//
__int64 __fastcall sub_14C03D0(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 a3,
        int a4,
        __int64 *a5,
        __int64 a6,
        void (__fastcall *a7)(unsigned __int64 *, __int64, __int64),
        __int64 a8,
        void (__fastcall *a9)(unsigned __int64 *, __int64, __int64, __int64),
        __int64 a10)
{
  unsigned int v10; // r12d
  __int64 v11; // r15
  __int64 v12; // r14
  unsigned int v13; // ebx
  unsigned __int64 v14; // rdx
  unsigned int v15; // r12d
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rax
  unsigned int v18; // edx
  bool v19; // cc
  __int64 result; // rax
  unsigned int v21; // ecx
  unsigned __int64 v22; // rax
  _QWORD *v23; // r14
  bool v24; // r15
  unsigned __int64 v25; // r15
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned __int64 v28; // rax
  __int64 **v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // r15
  __int64 v32; // rbx
  unsigned __int64 v33; // r12
  unsigned int v34; // ecx
  __int64 v35; // rcx
  unsigned int v36; // r15d
  int v37; // eax
  unsigned int v38; // edx
  __int64 v39; // rax
  char v40; // al
  __int64 v41; // rax
  unsigned int v42; // [rsp+10h] [rbp-A0h]
  unsigned int v43; // [rsp+10h] [rbp-A0h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  unsigned __int64 v45; // [rsp+20h] [rbp-90h]
  unsigned __int64 v47; // [rsp+28h] [rbp-88h]
  char v49; // [rsp+5Ah] [rbp-56h]
  char v50; // [rsp+5Bh] [rbp-55h]
  unsigned int v51; // [rsp+5Ch] [rbp-54h]
  _QWORD *v52; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+68h] [rbp-48h]
  unsigned __int64 v54; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v55; // [rsp+78h] [rbp-38h]

  v10 = *((_DWORD *)a2 + 2);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v11 = *(_QWORD *)(a1 - 8);
  else
    v11 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v12 = *(_QWORD *)(v11 + 24);
  v51 = a4 + 1;
  if ( *(_BYTE *)(v12 + 16) != 13 )
  {
    sub_14B86A0((__int64 *)v12, (__int64)a2, v51, a5);
    v21 = *((_DWORD *)a2 + 2);
    v53 = v21;
    if ( v21 > 0x40 )
    {
      sub_16A4FD0(&v52, a2);
      LOBYTE(v21) = v53;
      if ( v53 > 0x40 )
      {
        sub_16A8F40(&v52);
        v36 = v53;
        v23 = v52;
        v53 = 0;
        v55 = v36;
        v54 = (unsigned __int64)v52;
        if ( v36 > 0x40 )
        {
          v37 = sub_16A57B0(&v54);
          v38 = v36;
          v24 = 0;
          if ( v38 - v37 <= 0x40 )
            v24 = (unsigned __int64)v10 > *v23;
          if ( v23 )
          {
            j_j___libc_free_0_0(v23);
            if ( v53 > 0x40 )
            {
              if ( v52 )
              {
                j_j___libc_free_0_0(v52);
                v44 = (__int64)(a2 + 2);
                if ( v24 )
                  goto LABEL_21;
                goto LABEL_62;
              }
            }
          }
LABEL_20:
          v44 = (__int64)(a2 + 2);
          if ( v24 )
          {
LABEL_21:
            sub_16A5D10(&v54, a2, 64);
            v25 = v54;
            if ( v55 > 0x40 )
            {
              v25 = *(_QWORD *)v54;
              j_j___libc_free_0_0(v54);
            }
            sub_16A5D10(&v54, v44, 64);
            if ( v55 <= 0x40 )
            {
              v45 = v54;
            }
            else
            {
              v45 = *(_QWORD *)v54;
              j_j___libc_free_0_0(v54);
            }
            v26 = *((_DWORD *)a2 + 2);
            if ( v26 > 0x40 )
              memset((void *)*a2, 0, 8 * (((unsigned __int64)v26 + 63) >> 6));
            else
              *a2 = 0;
            v27 = *((_DWORD *)a2 + 6);
            if ( v27 > 0x40 )
              memset((void *)a2[2], 0, 8 * (((unsigned __int64)v27 + 63) >> 6));
            else
              a2[2] = 0;
            v28 = -1;
            if ( v10 )
              v28 = (((((((((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                        | (v10 - 1LL)
                        | (((unsigned __int64)v10 - 1) >> 1)) >> 4)
                      | (((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                      | (v10 - 1LL)
                      | (((unsigned __int64)v10 - 1) >> 1)) >> 8)
                    | (((((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                      | (v10 - 1LL)
                      | (((unsigned __int64)v10 - 1) >> 1)) >> 4)
                    | (((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                    | (v10 - 1LL)
                    | (((unsigned __int64)v10 - 1) >> 1)) >> 16)
                  | (((((((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                      | (v10 - 1LL)
                      | (((unsigned __int64)v10 - 1) >> 1)) >> 4)
                    | (((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                    | (v10 - 1LL)
                    | (((unsigned __int64)v10 - 1) >> 1)) >> 8)
                  | (((((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                    | (v10 - 1LL)
                    | (((unsigned __int64)v10 - 1) >> 1)) >> 4)
                  | (((v10 - 1LL) | (((unsigned __int64)v10 - 1) >> 1)) >> 2)
                  | (v10 - 1LL)
                  | (((unsigned __int64)v10 - 1) >> 1);
            v50 = 0;
            if ( (v28 & (v25 | v45)) == 0 )
            {
              v41 = sub_13CF970(a1);
              result = sub_14BE170(*(_QWORD *)(v41 + 24), v51, a5);
              if ( !(_BYTE)result )
                return result;
              v50 = 1;
            }
            if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              v29 = *(__int64 ***)(a1 - 8);
            else
              v29 = (__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
            sub_14B86A0(*v29, a3, v51, a5);
            sub_14A9CE0((__int64)a2);
            sub_14A9CE0(v44);
            v30 = ~v25;
            v49 = 1;
            v31 = 0;
            if ( v10 )
            {
              v32 = v10;
              v33 = v30;
              do
              {
                v34 = v31;
                if ( (v31 & v33) == v31 && (v31 | v45) == v31 )
                {
                  if ( v31 )
                    goto LABEL_39;
                  if ( !v50 )
                  {
                    v39 = sub_13CF970(a1);
                    v40 = sub_14BE170(*(_QWORD *)(v39 + 24), v51, a5);
                    v50 = 1;
                    v34 = 0;
                    v49 = v40;
                  }
                  if ( !v49 )
                  {
LABEL_39:
                    v42 = v34;
                    a7(&v54, a8, a3);
                    v35 = v42;
                    if ( *((_DWORD *)a2 + 2) > 0x40u )
                    {
                      sub_16A8890(a2, &v54);
                      v35 = v42;
                    }
                    else
                    {
                      *a2 &= v54;
                    }
                    if ( v55 > 0x40 && v54 )
                    {
                      v43 = v35;
                      j_j___libc_free_0_0(v54);
                      v35 = v43;
                    }
                    a9(&v54, a10, a3 + 16, v35);
                    if ( *((_DWORD *)a2 + 6) > 0x40u )
                      sub_16A8890(v44, &v54);
                    else
                      a2[2] &= v54;
                    if ( v55 > 0x40 && v54 )
                      j_j___libc_free_0_0(v54);
                  }
                }
                ++v31;
              }
              while ( v32 != v31 );
            }
            if ( *((_DWORD *)a2 + 2) <= 0x40u )
            {
              result = a2[2] & *a2;
              if ( !result )
                return result;
            }
            else
            {
              result = sub_16A59B0(a2, v44);
              if ( !(_BYTE)result )
                return result;
            }
            sub_14A9CE0((__int64)a2);
            return (__int64)sub_14A9DE0(v44);
          }
LABEL_62:
          sub_14A9DE0((__int64)a2);
          return (__int64)sub_14A9DE0(v44);
        }
LABEL_19:
        v24 = (unsigned __int64)v23 < v10;
        goto LABEL_20;
      }
      v22 = (unsigned __int64)v52;
    }
    else
    {
      v22 = *a2;
    }
    v23 = (_QWORD *)(~v22 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v21));
    goto LABEL_19;
  }
  v13 = *(_DWORD *)(v12 + 32);
  v14 = v10 - 1;
  v15 = v10 - 1;
  if ( v13 <= 0x40 )
  {
    if ( v14 >= *(_QWORD *)(v12 + 24) )
      v15 = *(_QWORD *)(v12 + 24);
  }
  else
  {
    v47 = v14;
    if ( v13 - (unsigned int)sub_16A57B0(v12 + 24) <= 0x40 && v47 >= **(_QWORD **)(v12 + 24) )
      v15 = **(_QWORD **)(v12 + 24);
  }
  sub_14B86A0(*(__int64 **)v11, (__int64)a2, v51, a5);
  ((void (__fastcall *)(unsigned __int64 *, __int64, unsigned __int64 *, _QWORD))a7)(&v54, a8, a2, v15);
  if ( *((_DWORD *)a2 + 2) > 0x40u && *a2 )
    j_j___libc_free_0_0(*a2);
  *a2 = v54;
  *((_DWORD *)a2 + 2) = v55;
  a9(&v54, a10, (__int64)(a2 + 2), v15);
  if ( *((_DWORD *)a2 + 6) > 0x40u )
  {
    v16 = a2[2];
    if ( v16 )
      j_j___libc_free_0_0(v16);
  }
  v17 = v54;
  v18 = v55;
  v19 = *((_DWORD *)a2 + 2) <= 0x40u;
  a2[2] = v54;
  *((_DWORD *)a2 + 6) = v18;
  if ( v19 )
  {
    result = *a2 & v17;
    if ( !result )
      return result;
  }
  else
  {
    result = sub_16A59B0(a2, a2 + 2);
    if ( !(_BYTE)result )
      return result;
  }
  sub_14A9CE0((__int64)a2);
  result = *((unsigned int *)a2 + 6);
  if ( (unsigned int)result > 0x40 )
    return (__int64)memset((void *)a2[2], 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
  a2[2] = 0;
  return result;
}
