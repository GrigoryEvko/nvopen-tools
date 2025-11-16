// Function: sub_9AB8E0
// Address: 0x9ab8e0
//
__int64 __fastcall sub_9AB8E0(unsigned __int8 *a1, __int64 a2, unsigned __int64 *a3, unsigned int a4, __m128i *a5)
{
  unsigned int v8; // r15d
  unsigned int *v9; // rax
  unsigned int v10; // eax
  unsigned __int8 v11; // al
  __int64 *v12; // rsi
  bool v13; // cc
  unsigned int v14; // eax
  unsigned __int64 v15; // rdi
  __int64 result; // rax
  unsigned int v17; // eax
  _BYTE *v18; // rax
  unsigned int v19; // r15d
  __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // r14d
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r15
  unsigned int v31; // edx
  __int64 v32; // rcx
  unsigned __int64 v33; // rcx
  int v34; // edx
  unsigned int *v35; // rax
  __int64 v36; // r8
  unsigned __int8 v37; // al
  __int64 v38; // rcx
  __int64 v39; // r8
  unsigned __int8 v40; // al
  __int64 v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+10h] [rbp-A0h]
  int v44; // [rsp+18h] [rbp-98h]
  int v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v47; // [rsp+28h] [rbp-88h]
  unsigned __int64 v48; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v49; // [rsp+38h] [rbp-78h]
  __int64 v50[2]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v51; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v52; // [rsp+58h] [rbp-58h]
  unsigned __int64 v53; // [rsp+60h] [rbp-50h]
  unsigned int v54; // [rsp+68h] [rbp-48h]
  char v55; // [rsp+70h] [rbp-40h]

  v8 = *(_DWORD *)(a2 + 8);
  if ( v8 > 0x40 )
  {
    if ( v8 != (unsigned int)sub_C444A0(a2) )
      goto LABEL_3;
LABEL_16:
    v17 = *((_DWORD *)a3 + 2);
    if ( v17 > 0x40 )
      memset((void *)*a3, 0, 8 * (((unsigned __int64)v17 + 63) >> 6));
    else
      *a3 = 0;
    goto LABEL_18;
  }
  if ( !*(_QWORD *)a2 )
    goto LABEL_16;
LABEL_3:
  v9 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v9 )
    v10 = *v9;
  else
    v10 = qword_4F862D0[2];
  if ( a4 > v10 )
    goto LABEL_16;
  v11 = *a1;
  v12 = (__int64 *)(a1 + 24);
  if ( *a1 == 17 )
  {
LABEL_7:
    sub_987BA0((__int64)&v51, v12);
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
      j_j___libc_free_0_0(*a3);
    v13 = *((_DWORD *)a3 + 6) <= 0x40u;
    *a3 = v51;
    v14 = v52;
    v52 = 0;
    *((_DWORD *)a3 + 2) = v14;
    if ( v13 || (v15 = a3[2]) == 0 )
    {
      a3[2] = v53;
      result = v54;
      *((_DWORD *)a3 + 6) = v54;
    }
    else
    {
      j_j___libc_free_0_0(v15);
      v13 = v52 <= 0x40;
      a3[2] = v53;
      result = v54;
      *((_DWORD *)a3 + 6) = v54;
      if ( !v13 )
      {
        if ( v51 )
          return j_j___libc_free_0_0(v51);
      }
    }
    return result;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a1 + 1) + 8LL) - 17 <= 1 )
  {
    if ( v11 > 0x15u )
    {
      v42 = (__int64)(a3 + 2);
      goto LABEL_89;
    }
    v18 = (_BYTE *)sub_AD7630(a1, 0);
    if ( v18 )
    {
      v12 = (__int64 *)(v18 + 24);
      if ( *v18 == 17 )
        goto LABEL_7;
    }
    v11 = *a1;
  }
  if ( v11 == 20 || v11 == 14 )
  {
    sub_986FF0((__int64)a3);
LABEL_18:
    result = *((unsigned int *)a3 + 6);
    if ( (unsigned int)result > 0x40 )
      return (__int64)memset((void *)a3[2], 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
    a3[2] = 0;
    return result;
  }
  v42 = (__int64)(a3 + 2);
  if ( v11 == 16 )
  {
    v19 = 0;
    sub_986FF0((__int64)a3);
    sub_986FF0(v42);
    v44 = sub_AC5290(a1);
    if ( !v44 )
    {
LABEL_83:
      if ( *((_DWORD *)a3 + 2) <= 0x40u )
      {
        result = a3[2] & *a3;
        if ( !result )
          return result;
      }
      else
      {
        result = sub_C446A0(a3, v42);
        if ( !(_BYTE)result )
          return result;
      }
LABEL_85:
      sub_987100((__int64)a3);
      return (__int64)sub_987100(v42);
    }
    while ( 1 )
    {
      v22 = *(_QWORD *)a2;
      if ( *(_DWORD *)(a2 + 8) > 0x40u )
        v22 = *(_QWORD *)(v22 + 8LL * (v19 >> 6));
      if ( (v22 & (1LL << v19)) == 0 )
        goto LABEL_51;
      sub_AC5390(&v46, a1, v19);
      v23 = v47;
      v49 = v47;
      if ( v47 <= 0x40 )
        break;
      sub_C43780(&v48, &v46);
      v23 = v49;
      if ( v49 <= 0x40 )
      {
        v20 = v48;
        goto LABEL_35;
      }
      sub_C43D10(&v48, &v46, v24, v25, v26);
      v21 = v48;
      v23 = v49;
LABEL_38:
      v13 = *((_DWORD *)a3 + 2) <= 0x40u;
      v52 = v23;
      v51 = v21;
      v49 = 0;
      if ( v13 )
      {
        *a3 &= v21;
      }
      else
      {
        sub_C43B90(a3, &v51);
        v23 = v52;
      }
      if ( v23 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( *((_DWORD *)a3 + 6) > 0x40u )
        sub_C43B90(v42, &v46);
      else
        a3[2] &= v46;
      if ( v47 > 0x40 && v46 )
        j_j___libc_free_0_0(v46);
LABEL_51:
      if ( v44 == ++v19 )
        goto LABEL_83;
    }
    v20 = v46;
LABEL_35:
    v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & ~v20;
    if ( !v23 )
      v21 = 0;
    v48 = v21;
    goto LABEL_38;
  }
  if ( v11 == 11 )
  {
    v27 = 0;
    sub_986FF0((__int64)a3);
    sub_986FF0(v42);
    v45 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
    if ( !v45 )
      goto LABEL_83;
    while ( 1 )
    {
      v28 = *(_QWORD *)a2;
      if ( *(_DWORD *)(a2 + 8) > 0x40u )
        v28 = *(_QWORD *)(v28 + 8LL * (v27 >> 6));
      if ( (v28 & (1LL << v27)) == 0 )
        goto LABEL_82;
      v29 = sub_AD69F0(a1, v27);
      v30 = v29;
      if ( *(_BYTE *)v29 == 13 )
        goto LABEL_82;
      if ( *(_BYTE *)v29 != 17 )
        goto LABEL_85;
      v31 = *(_DWORD *)(v29 + 32);
      v41 = v29 + 24;
      v49 = v31;
      if ( v31 <= 0x40 )
        break;
      sub_C43780(&v48, v41);
      v31 = v49;
      if ( v49 <= 0x40 )
      {
        v32 = v48;
        goto LABEL_69;
      }
      sub_C43D10(&v48, v41, v49, v38, v39);
      v31 = v49;
      v33 = v48;
LABEL_72:
      v13 = *((_DWORD *)a3 + 2) <= 0x40u;
      v52 = v31;
      v51 = v33;
      v49 = 0;
      if ( v13 )
      {
        *a3 &= v33;
      }
      else
      {
        sub_C43B90(a3, &v51);
        v31 = v52;
      }
      if ( v31 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( *((_DWORD *)a3 + 6) > 0x40u )
        sub_C43B90(v42, v41);
      else
        a3[2] &= *(_QWORD *)(v30 + 24);
LABEL_82:
      if ( ++v27 == v45 )
        goto LABEL_83;
    }
    v32 = *(_QWORD *)(v29 + 24);
LABEL_69:
    v33 = ~v32 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31);
    if ( !v31 )
      v33 = 0;
    v48 = v33;
    goto LABEL_72;
  }
LABEL_89:
  sub_987100((__int64)a3);
  sub_987100(v42);
  v34 = *a1;
  result = (unsigned int)(v34 - 12);
  if ( (unsigned int)result > 1 )
  {
    if ( (_BYTE)v34 == 22 )
    {
      sub_B2D8F0(&v51, a1);
      if ( v55 )
      {
        sub_AB0A90(&v48, &v51);
        sub_984AC0((__int64 *)a3, (__int64 *)&v48);
        sub_969240(v50);
        sub_969240((__int64 *)&v48);
        if ( v55 )
        {
          v55 = 0;
          if ( v54 > 0x40 && v53 )
            j_j___libc_free_0_0(v53);
          if ( v52 > 0x40 && v51 )
            j_j___libc_free_0_0(v51);
        }
      }
    }
    v35 = (unsigned int *)sub_C94E20(qword_4F862D0);
    if ( v35 )
      result = *v35;
    else
      result = LODWORD(qword_4F862D0[2]);
    if ( a4 != (_DWORD)result )
    {
      v37 = *a1;
      if ( *a1 == 1 )
      {
        result = sub_B2F6B0(a1);
        if ( !(_BYTE)result )
          return sub_9AC0E0(*((_QWORD *)a1 - 4), a3, a4 + 1, a5);
      }
      else
      {
        if ( v37 == 5 || v37 > 0x1Cu )
        {
          sub_9A7810(a1, a2, (__int64)a3, a4, a5);
        }
        else if ( (unsigned __int8)(v37 - 2) <= 1u || !v37 )
        {
          sub_B2FDE0(&v51, a1);
          if ( v55 )
          {
            sub_AB0A90(&v48, &v51);
            sub_984AC0((__int64 *)a3, (__int64 *)&v48);
            sub_969240(v50);
            sub_969240((__int64 *)&v48);
            if ( v55 )
              sub_9963D0((__int64)&v51);
          }
        }
        if ( *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) == 14 )
        {
          v40 = sub_BD5420(a1, a5->m128i_i64[0]);
          sub_9870B0((__int64)a3, 0, v40);
        }
        return sub_99B5E0((__int64)a1, (__int64)a3, a4, a5->m128i_i64, v36);
      }
    }
  }
  return result;
}
