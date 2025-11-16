// Function: sub_28F4ED0
// Address: 0x28f4ed0
//
__int64 __fastcall sub_28F4ED0(__int64 a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  unsigned int v7; // r15d
  __int64 v8; // r14
  __int64 v10; // r11
  __int16 v11; // r10
  __int64 v12; // rbx
  __int64 *v13; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // cl
  char v18; // dl
  char v19; // al
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 *v23; // r8
  unsigned __int64 v24; // rdi
  __int64 *v25; // rax
  const void **v26; // rsi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  unsigned int v29; // esi
  int v30; // eax
  bool v31; // al
  __int16 v32; // kr00_2
  int v33; // eax
  int v34; // eax
  unsigned int v35; // esi
  int v36; // eax
  bool v37; // al
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  int v40; // eax
  int v41; // eax
  char v42; // [rsp+8h] [rbp-A8h]
  char v43; // [rsp+8h] [rbp-A8h]
  char v44; // [rsp+8h] [rbp-A8h]
  char v45; // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  __int64 v47; // [rsp+10h] [rbp-A0h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  __int64 v49; // [rsp+18h] [rbp-98h]
  char v50; // [rsp+18h] [rbp-98h]
  char v51; // [rsp+18h] [rbp-98h]
  char v52; // [rsp+18h] [rbp-98h]
  char v53; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+20h] [rbp-90h]
  __int64 *v55; // [rsp+20h] [rbp-90h]
  const void **v56; // [rsp+28h] [rbp-88h]
  char v57; // [rsp+28h] [rbp-88h]
  char v58; // [rsp+28h] [rbp-88h]
  int v60; // [rsp+38h] [rbp-78h]
  __int64 *v61; // [rsp+38h] [rbp-78h]
  unsigned __int64 v62; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v63; // [rsp+48h] [rbp-68h]
  unsigned __int64 v64; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v65; // [rsp+58h] [rbp-58h]
  unsigned __int64 v66; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v67; // [rsp+68h] [rbp-48h]

  v7 = 0;
  v8 = *(_QWORD *)(a4 + 8);
  if ( v8 != *(_QWORD *)(a5 + 8) )
    return v7;
  v60 = 1;
  v10 = a2;
  v11 = a3;
  v12 = a4;
  v13 = (__int64 *)a5;
  v15 = *(_QWORD *)(*(_QWORD *)a4 + 16LL);
  if ( v15 )
    v60 = (*(_QWORD *)(v15 + 8) == 0) + 1;
  v16 = *(_QWORD *)(*(_QWORD *)a5 + 16LL);
  if ( v16 )
    v60 += *(_QWORD *)(v16 + 8) == 0;
  v17 = *(_BYTE *)(a4 + 36);
  v18 = HIBYTE(a3);
  v19 = *(_BYTE *)(a5 + 36);
  if ( v17 == v19 )
  {
    v7 = *(_DWORD *)(v12 + 24);
    v26 = (const void **)(v12 + 16);
    v67 = v7;
    if ( v17 )
    {
      if ( v7 > 0x40 )
      {
        v44 = v11;
        v48 = v10;
        v53 = HIBYTE(v11);
        v55 = (__int64 *)(a5 + 16);
        sub_C43780((__int64)&v66, v26);
        v7 = v67;
        v18 = v53;
        v10 = v48;
        LOBYTE(v11) = v44;
        if ( v67 > 0x40 )
        {
          sub_C43C10(&v66, v55);
          v7 = v67;
          v28 = v66;
          v18 = v53;
          v10 = v48;
          v65 = v67;
          LOBYTE(v11) = v44;
          v64 = v66;
          if ( v67 > 0x40 )
          {
            v40 = sub_C444A0((__int64)&v64);
            v18 = v53;
            v10 = v48;
            LOBYTE(v11) = v44;
            if ( v7 != v40 )
            {
              v41 = sub_C445E0((__int64)&v64);
              v23 = (__int64 *)&v64;
              v18 = v53;
              v10 = v48;
              LOBYTE(v11) = v44;
              LOBYTE(v7) = v7 == v41;
              goto LABEL_34;
            }
LABEL_39:
            *a7 = sub_28EA8C0(v10, v11, v18, v8, (__int64)&v64);
            if ( *(_DWORD *)(a6 + 8) > 0x40u )
              sub_C43C10((_QWORD *)a6, (__int64 *)&v64);
            else
              *(_QWORD *)a6 ^= v64;
LABEL_59:
            if ( v65 <= 0x40 )
              goto LABEL_22;
            v24 = v64;
            if ( !v64 )
              goto LABEL_22;
            goto LABEL_21;
          }
LABEL_31:
          if ( v28 && v7 )
          {
            v23 = (__int64 *)&v64;
            LOBYTE(v7) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == v28;
LABEL_34:
            if ( !(_BYTE)v7 )
            {
              v29 = *(_DWORD *)(a6 + 8);
              if ( v29 <= 0x40 )
              {
                v31 = *(_QWORD *)a6 == 0;
              }
              else
              {
                v45 = v11;
                v49 = v10;
                v57 = v18;
                v30 = sub_C444A0(a6);
                v23 = (__int64 *)&v64;
                LOBYTE(v11) = v45;
                v10 = v49;
                v18 = v57;
                v31 = v29 == v30;
              }
              if ( v60 <= 1 && v31 )
                goto LABEL_54;
            }
            goto LABEL_39;
          }
          goto LABEL_39;
        }
        v27 = v66;
      }
      else
      {
        v27 = *(_QWORD *)(v12 + 16);
      }
      v28 = v13[2] ^ v27;
      v65 = v7;
      v64 = v28;
      goto LABEL_31;
    }
    if ( v7 > 0x40 )
    {
      v52 = v11;
      v54 = v10;
      v58 = HIBYTE(v11);
      v61 = (__int64 *)(a5 + 16);
      sub_C43780((__int64)&v66, v26);
      v7 = v67;
      v18 = v58;
      v10 = v54;
      LOBYTE(v11) = v52;
      if ( v67 > 0x40 )
      {
        sub_C43C10(&v66, v61);
        v7 = v67;
        v39 = v66;
        LOBYTE(v11) = v52;
        v10 = v54;
        v18 = v58;
        goto LABEL_58;
      }
      v38 = v66;
    }
    else
    {
      v38 = *(_QWORD *)(v12 + 16);
    }
    v39 = v13[2] ^ v38;
LABEL_58:
    v65 = v7;
    v64 = v39;
    *a7 = sub_28EA8C0(v10, v11, v18, v8, (__int64)&v64);
    goto LABEL_59;
  }
  if ( v19 )
  {
    v25 = (__int64 *)v12;
    v12 = a5;
    v13 = v25;
  }
  v7 = *(_DWORD *)(v12 + 24);
  v56 = (const void **)(v12 + 16);
  v65 = v7;
  if ( v7 <= 0x40 )
  {
    v20 = *(_QWORD *)(v12 + 16);
    goto LABEL_12;
  }
  v32 = v11;
  sub_C43780((__int64)&v64, v56);
  v7 = v65;
  v18 = HIBYTE(v32);
  v10 = a2;
  LOBYTE(v11) = v32;
  if ( v65 <= 0x40 )
  {
    v20 = v64;
LABEL_12:
    v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v20;
    if ( !v7 )
      v21 = 0;
    goto LABEL_14;
  }
  sub_C43D10((__int64)&v64);
  v7 = v65;
  v21 = v64;
  v65 = 0;
  v18 = HIBYTE(v32);
  v10 = a2;
  v67 = v7;
  LOBYTE(v11) = v32;
  v66 = v64;
  if ( v7 <= 0x40 )
  {
LABEL_14:
    v22 = v13[2] ^ v21;
    v63 = v7;
    v62 = v22;
LABEL_15:
    v23 = (__int64 *)&v62;
    if ( !v62 || !v7 )
      goto LABEL_17;
    LOBYTE(v7) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == v62;
    goto LABEL_49;
  }
  sub_C43C10(&v66, v13 + 2);
  v7 = v67;
  v18 = HIBYTE(v32);
  v63 = v67;
  v10 = a2;
  v62 = v66;
  LOBYTE(v11) = v32;
  if ( v65 > 0x40 && v64 )
  {
    j_j___libc_free_0_0(v64);
    v7 = v63;
    LOBYTE(v11) = v32;
    v10 = a2;
    v18 = HIBYTE(v32);
  }
  if ( v7 <= 0x40 )
    goto LABEL_15;
  v42 = v11;
  v46 = v10;
  v50 = v18;
  v33 = sub_C444A0((__int64)&v62);
  v18 = v50;
  v10 = v46;
  LOBYTE(v11) = v42;
  if ( v7 == v33 )
    goto LABEL_17;
  v34 = sub_C445E0((__int64)&v62);
  v23 = (__int64 *)&v62;
  v18 = v50;
  v10 = v46;
  LOBYTE(v11) = v42;
  LOBYTE(v7) = v7 == v34;
LABEL_49:
  if ( !(_BYTE)v7 )
  {
    v35 = *(_DWORD *)(a6 + 8);
    if ( v35 <= 0x40 )
    {
      v37 = *(_QWORD *)a6 == 0;
    }
    else
    {
      v43 = v11;
      v47 = v10;
      v51 = v18;
      v36 = sub_C444A0(a6);
      v23 = (__int64 *)&v62;
      LOBYTE(v11) = v43;
      v10 = v47;
      v18 = v51;
      v37 = v35 == v36;
    }
    if ( v60 <= 1 && v37 )
    {
LABEL_54:
      sub_969240(v23);
      return v7;
    }
  }
LABEL_17:
  *a7 = sub_28EA8C0(v10, v11, v18, v8, (__int64)&v62);
  if ( *(_DWORD *)(a6 + 8) > 0x40u )
    sub_C43C10((_QWORD *)a6, (__int64 *)v56);
  else
    *(_QWORD *)a6 ^= *(_QWORD *)(v12 + 16);
  if ( v63 <= 0x40 )
    goto LABEL_22;
  v24 = v62;
  if ( !v62 )
    goto LABEL_22;
LABEL_21:
  j_j___libc_free_0_0(v24);
LABEL_22:
  if ( **(_BYTE **)v12 > 0x1Cu )
  {
    sub_D68D20((__int64)&v66, 0, *(_QWORD *)v12);
    sub_28F19A0(a1 + 64, &v66);
    sub_D68D70(&v66);
  }
  v7 = 1;
  if ( *(_BYTE *)*v13 > 0x1Cu )
  {
    sub_D68D20((__int64)&v66, 0, *v13);
    sub_28F19A0(a1 + 64, &v66);
    sub_D68D70(&v66);
  }
  return v7;
}
