// Function: sub_101AA20
// Address: 0x101aa20
//
__int64 __fastcall sub_101AA20(unsigned int a1, unsigned __int8 *a2, unsigned __int8 *a3, __m128i *a4, unsigned int a5)
{
  __int64 v7; // r12
  __int64 v9; // rsi
  unsigned __int8 *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  _BYTE *v13; // rdx
  unsigned __int64 v14; // rcx
  __int64 v15; // rdi
  int v16; // ecx
  __int64 v17; // rbx
  unsigned int v18; // ebx
  char v19; // r13
  int v20; // edx
  int v21; // edx
  _BYTE *v22; // rax
  __int64 v23; // rdi
  int v24; // esi
  __int64 v25; // r15
  unsigned int v26; // ebx
  char v27; // bl
  __int64 v28; // r13
  __int64 v29; // rdx
  _BYTE *v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rdx
  _BYTE *v33; // rax
  int v34; // eax
  unsigned int v35; // r15d
  _BYTE *v36; // rax
  int v37; // eax
  unsigned int v38; // r14d
  _BYTE *v39; // rax
  int v40; // [rsp+Ch] [rbp-64h]
  int v41; // [rsp+Ch] [rbp-64h]
  unsigned __int8 *v42; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int8 *v43; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v44; // [rsp+28h] [rbp-48h] BYREF
  __int64 v45; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+38h] [rbp-38h]

  v43 = a2;
  v42 = a3;
  v7 = sub_FFE3E0(a1, &v43, &v42, a4->m128i_i64);
  if ( v7 )
    return v7;
  v9 = (__int64)v43;
  v7 = (__int64)sub_1019820(a1, (__int64)v43, v42, a4, a5);
  if ( v7 || !a4[4].m128i_i8[0] )
    return v7;
  v10 = v43;
  v11 = *v43;
  if ( a1 == 23 )
  {
    if ( (unsigned __int8)v11 <= 0x1Cu )
    {
      if ( (_BYTE)v11 != 5 )
        goto LABEL_8;
      v21 = *((unsigned __int16 *)v43 + 1);
      if ( (*((_WORD *)v43 + 1) & 0xFFF7) != 0x11 && (v21 & 0xFFFD) != 0xD )
        goto LABEL_8;
    }
    else
    {
      if ( (unsigned __int8)v11 > 0x36u )
        goto LABEL_8;
      v9 = 0x40540000000000LL;
      if ( !_bittest64(&v9, v11) )
        goto LABEL_8;
      v21 = (unsigned __int8)v11 - 29;
    }
    if ( v21 == 25 && (v43[1] & 4) != 0 )
      goto LABEL_29;
  }
  else
  {
    if ( (unsigned __int8)v11 > 0x1Cu )
    {
      if ( (unsigned __int8)v11 > 0x36u )
        goto LABEL_8;
      v9 = 0x40540000000000LL;
      if ( !_bittest64(&v9, v11) )
        goto LABEL_8;
      v20 = (unsigned __int8)v11 - 29;
    }
    else
    {
      if ( (_BYTE)v11 != 5 )
        goto LABEL_8;
      v20 = *((unsigned __int16 *)v43 + 1);
      if ( (*((_WORD *)v43 + 1) & 0xFFFD) != 0xD && (v20 & 0xFFF7) != 0x11 )
        goto LABEL_8;
    }
    if ( v20 == 25 && (v43[1] & 2) != 0 )
    {
LABEL_29:
      if ( *((unsigned __int8 **)v43 - 8) == v42 )
        return sub_AD6530(*((_QWORD *)v10 + 1), v9);
    }
  }
LABEL_8:
  v12 = *v42;
  if ( (_BYTE)v12 == 17 )
  {
    v13 = v42 + 24;
  }
  else
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v42 + 1) + 8LL) - 17 > 1 )
      return v7;
    if ( (unsigned __int8)v12 > 0x15u )
      return v7;
    v22 = sub_AD7630((__int64)v42, 0, v12);
    if ( !v22 || *v22 != 17 )
      return v7;
    v13 = v22 + 24;
    v10 = v43;
  }
  v44 = v13;
  v14 = *v10;
  if ( a1 == 23 )
  {
    if ( (unsigned __int8)v14 <= 0x1Cu )
    {
      if ( (_BYTE)v14 != 5 )
        return v7;
      v24 = *((unsigned __int16 *)v10 + 1);
      if ( (*((_WORD *)v10 + 1) & 0xFFF7) != 0x11 && (v24 & 0xFFFD) != 0xD )
        return v7;
    }
    else
    {
      if ( (unsigned __int8)v14 > 0x36u )
        return v7;
      v23 = 0x40540000000000LL;
      v24 = (unsigned __int8)v14 - 29;
      if ( !_bittest64(&v23, v14) )
        return v7;
    }
    if ( v24 == 17 && (v10[1] & 4) != 0 )
    {
      v25 = *((_QWORD *)v10 - 4);
      if ( *(_BYTE *)v25 == 17 )
      {
        v9 = v25 + 24;
        sub_C4B8A0((__int64)&v45, v25 + 24, (__int64)v13);
        v26 = v46;
        if ( v46 <= 0x40 )
        {
          v27 = v45 == 0;
        }
        else
        {
          v27 = v26 == (unsigned int)sub_C444A0((__int64)&v45);
          if ( v45 )
            j_j___libc_free_0_0(v45);
        }
      }
      else
      {
        v31 = *(_QWORD *)(v25 + 8);
        v32 = (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17;
        if ( (unsigned int)v32 > 1 || *(_BYTE *)v25 > 0x15u )
          return v7;
        v9 = 0;
        v33 = sub_AD7630(v25, 0, v32);
        if ( v33 && *v33 == 17 )
        {
          v9 = (__int64)(v33 + 24);
          v27 = sub_FFF1E0((__int64 *)&v44, (__int64)(v33 + 24));
        }
        else
        {
          if ( *(_BYTE *)(v31 + 8) != 17 )
            return v7;
          v37 = *(_DWORD *)(v31 + 32);
          v38 = 0;
          v27 = 0;
          v41 = v37;
          while ( v41 != v38 )
          {
            v9 = v38;
            v39 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v25, v38);
            if ( !v39 )
              return v7;
            if ( *v39 != 13 )
            {
              if ( *v39 != 17 )
                return v7;
              v9 = (__int64)(v39 + 24);
              v27 = sub_FFF1E0((__int64 *)&v44, (__int64)(v39 + 24));
              if ( !v27 )
                return v7;
            }
            ++v38;
          }
        }
      }
      if ( !v27 )
        return v7;
      goto LABEL_22;
    }
  }
  else
  {
    if ( (unsigned __int8)v14 <= 0x1Cu )
    {
      if ( (_BYTE)v14 != 5 )
        return v7;
      v16 = *((unsigned __int16 *)v10 + 1);
      if ( (*((_WORD *)v10 + 1) & 0xFFF7) != 0x11 && (v16 & 0xFFFD) != 0xD )
        return v7;
    }
    else
    {
      if ( (unsigned __int8)v14 > 0x36u )
        return v7;
      v15 = 0x40540000000000LL;
      if ( !_bittest64(&v15, v14) )
        return v7;
      v16 = (unsigned __int8)v14 - 29;
    }
    if ( v16 == 17 && (v10[1] & 2) != 0 )
    {
      v17 = *((_QWORD *)v10 - 4);
      if ( *(_BYTE *)v17 == 17 )
      {
        v9 = v17 + 24;
        sub_C4B490((__int64)&v45, v17 + 24, (__int64)v13);
        v18 = v46;
        if ( v46 <= 0x40 )
          v19 = v45 == 0;
        else
          v19 = v18 == (unsigned int)sub_C444A0((__int64)&v45);
        sub_969240(&v45);
      }
      else
      {
        v28 = *(_QWORD *)(v17 + 8);
        v29 = (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17;
        if ( (unsigned int)v29 > 1 || *(_BYTE *)v17 > 0x15u )
          return v7;
        v9 = 0;
        v30 = sub_AD7630(v17, 0, v29);
        if ( v30 && *v30 == 17 )
        {
          v9 = (__int64)(v30 + 24);
          v19 = sub_FFF170((__int64 *)&v44, (__int64)(v30 + 24));
        }
        else
        {
          if ( *(_BYTE *)(v28 + 8) != 17 )
            return v7;
          v34 = *(_DWORD *)(v28 + 32);
          v35 = 0;
          v19 = 0;
          v40 = v34;
          while ( v40 != v35 )
          {
            v9 = v35;
            v36 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v17, v35);
            if ( !v36 )
              return v7;
            if ( *v36 != 13 )
            {
              if ( *v36 != 17 )
                return v7;
              v9 = (__int64)(v36 + 24);
              v19 = sub_FFF170((__int64 *)&v44, (__int64)(v36 + 24));
              if ( !v19 )
                return v7;
            }
            ++v35;
          }
        }
      }
      if ( v19 )
      {
LABEL_22:
        v10 = v43;
        return sub_AD6530(*((_QWORD *)v10 + 1), v9);
      }
    }
  }
  return v7;
}
