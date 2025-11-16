// Function: sub_C48A30
// Address: 0xc48a30
//
__int64 __fastcall sub_C48A30(__int64 a1, _QWORD *a2, unsigned int a3, char a4, bool a5, char a6, char a7)
{
  const void **v7; // r11
  unsigned __int64 v10; // rbx
  char v11; // r15
  char *v12; // r14
  unsigned int v13; // edx
  int v14; // eax
  bool v15; // al
  __int64 result; // rax
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r8
  unsigned int v20; // esi
  _BYTE *v21; // r14
  _BYTE *v22; // rcx
  __int64 v23; // r11
  unsigned int v24; // r14d
  unsigned int v25; // eax
  unsigned int v26; // edx
  unsigned int v27; // ebx
  _QWORD *v28; // rax
  __int64 v29; // r12
  _QWORD *v30; // r15
  int v31; // eax
  unsigned __int64 *v32; // rax
  __int64 v33; // r9
  char v34; // r9
  size_t v35; // rbx
  _BYTE *v36; // rdx
  unsigned int v37; // esi
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  unsigned int v40; // r14d
  __int64 v41; // rcx
  char v42; // r13
  unsigned int v43; // r13d
  char *v44; // r11
  char v45; // dl
  char v46; // cl
  __int64 *v47; // rax
  _QWORD *v48; // [rsp+0h] [rbp-C0h]
  unsigned int v49; // [rsp+8h] [rbp-B8h]
  const void **v50; // [rsp+8h] [rbp-B8h]
  unsigned int v51; // [rsp+8h] [rbp-B8h]
  unsigned int v53; // [rsp+10h] [rbp-B0h]
  unsigned int v54; // [rsp+10h] [rbp-B0h]
  bool v55; // [rsp+18h] [rbp-A8h]
  unsigned int v56; // [rsp+18h] [rbp-A8h]
  char v57; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  unsigned int v59; // [rsp+18h] [rbp-A8h]
  __int64 v60; // [rsp+20h] [rbp-A0h]
  unsigned int v61; // [rsp+28h] [rbp-98h]
  unsigned __int64 v62; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int64 v63; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v64; // [rsp+48h] [rbp-78h]
  _BYTE v65[63]; // [rsp+81h] [rbp-3Fh] BYREF

  v7 = (const void **)a1;
  v10 = a3;
  v55 = a5;
  if ( a5 )
  {
    if ( a3 == 10 )
    {
      v11 = 0;
      v12 = (char *)byte_3F871B3;
      goto LABEL_7;
    }
    if ( a3 > 0xA )
    {
      if ( a3 != 16 )
        goto LABEL_129;
      v12 = "0x";
    }
    else
    {
      if ( a3 != 2 )
      {
        if ( a3 == 8 )
        {
          v11 = 48;
          v12 = "0";
LABEL_7:
          v61 = 3;
          v55 = a3 == 8;
          goto LABEL_8;
        }
LABEL_129:
        BUG();
      }
      v12 = "0b";
    }
    v13 = *(_DWORD *)(a1 + 8);
    v11 = 48;
    v61 = 4;
    v55 = (_DWORD)v10 == 8;
    if ( v13 > 0x40 )
      goto LABEL_9;
LABEL_32:
    v15 = *(_QWORD *)a1 == 0;
    goto LABEL_10;
  }
  if ( (a3 & 0xFFFFFFFD) == 8 )
  {
    v12 = (char *)byte_3F871B3;
    v61 = 3;
    v55 = a3 == 8;
    v11 = 0;
  }
  else
  {
    v61 = 4;
    v11 = 0;
    v12 = (char *)byte_3F871B3;
  }
LABEL_8:
  v13 = *(_DWORD *)(a1 + 8);
  if ( v13 <= 0x40 )
    goto LABEL_32;
LABEL_9:
  v49 = v13;
  v14 = sub_C444A0(a1);
  v13 = v49;
  v7 = (const void **)a1;
  v15 = v49 == v14;
LABEL_10:
  if ( v15 )
  {
    for ( result = a2[1]; v11; a2[1] = result )
    {
      if ( (unsigned __int64)(result + 1) > a2[2] )
      {
        sub_C8D290(a2, a2 + 3, result + 1, 1);
        result = a2[1];
      }
      ++v12;
      *(_BYTE *)(*a2 + result) = v11;
      v11 = *v12;
      result = a2[1] + 1LL;
    }
    if ( (unsigned __int64)(result + 1) > a2[2] )
    {
      sub_C8D290(a2, a2 + 3, result + 1, 1);
      result = a2[1];
    }
    *(_BYTE *)(*a2 + result) = 48;
    ++a2[1];
    return result;
  }
  result = (__int64)&a0123456789abcd_6[-36];
  if ( a6 )
    result = (__int64)"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  v60 = result;
  if ( v13 <= 0x40 )
  {
    v17 = (unsigned __int64)*v7;
    v18 = a2[1];
    if ( !a4 )
      goto LABEL_15;
    if ( v13 )
    {
      result = (__int64)(v17 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
      v17 = result;
      if ( result >= 0 )
      {
LABEL_15:
        if ( !v11 )
          goto LABEL_20;
        goto LABEL_16;
      }
      if ( v18 + 1 > a2[2] )
      {
        v58 = result;
        sub_C8D290(a2, a2 + 3, v18 + 1, 1);
        v18 = a2[1];
        result = v58;
      }
      v17 = -result;
      *(_BYTE *)(*a2 + v18) = 45;
      v18 = a2[1] + 1LL;
      a2[1] = v18;
      if ( !v11 )
      {
        v19 = a2[2];
        goto LABEL_21;
      }
    }
    else
    {
      v17 = 0;
      if ( !v11 )
      {
        v19 = a2[2];
        goto LABEL_116;
      }
    }
    do
    {
LABEL_16:
      if ( v18 + 1 > a2[2] )
      {
        sub_C8D290(a2, a2 + 3, v18 + 1, 1);
        v18 = a2[1];
      }
      ++v12;
      *(_BYTE *)(*a2 + v18) = v11;
      result = a2[1];
      v11 = *v12;
      v18 = result + 1;
      a2[1] = result + 1;
    }
    while ( v11 );
    v10 = (unsigned int)v10;
LABEL_20:
    v19 = a2[2];
    if ( v17 )
    {
LABEL_21:
      v20 = 0;
      v21 = v65;
      while ( 1 )
      {
        v22 = v21;
        if ( a7 )
        {
          if ( v20 % v61 || !v20 )
          {
            v22 = v21;
          }
          else
          {
            *(v21 - 1) = 39;
            v22 = v21 - 1;
          }
        }
        v21 = v22 - 1;
        ++v20;
        result = v17 / v10;
        *(v22 - 1) = *(_BYTE *)(v60 + v17 % v10);
        if ( v17 < v10 )
          break;
        v17 /= v10;
      }
      v35 = v65 - v21;
      v36 = (_BYTE *)(v65 - v21 + v18);
      if ( (unsigned __int64)v36 <= v19 )
        goto LABEL_73;
      goto LABEL_118;
    }
LABEL_116:
    if ( v19 >= v18 )
    {
LABEL_75:
      a2[1] = v18;
      return result;
    }
    v36 = (_BYTE *)v18;
    v35 = 0;
    v21 = v65;
LABEL_118:
    result = sub_C8D290(a2, a2 + 3, v36, 1);
    v18 = a2[1];
LABEL_73:
    if ( v21 == v65 )
    {
      v18 += v35;
    }
    else
    {
      result = (__int64)memcpy((void *)(*a2 + v18), v21, v35);
      v18 = v35 + a2[1];
    }
    goto LABEL_75;
  }
  v50 = v7;
  v64 = v13;
  sub_C43780((__int64)&v63, v7);
  if ( !a4 )
    goto LABEL_36;
  v37 = *((_DWORD *)v50 + 2);
  v38 = (unsigned __int64)*v50;
  if ( v37 > 0x40 )
    v38 = *(_QWORD *)(v38 + 8LL * ((v37 - 1) >> 6));
  if ( (v38 & (1LL << ((unsigned __int8)v37 - 1))) != 0 )
  {
    if ( v64 <= 0x40 )
    {
      v47 = (__int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v64) & ~v63);
      if ( !v64 )
        v47 = 0;
      v63 = (unsigned __int64)v47;
    }
    else
    {
      sub_C43D10((__int64)&v63);
    }
    sub_C46250((__int64)&v63);
    v39 = a2[1];
    if ( (unsigned __int64)(v39 + 1) > a2[2] )
    {
      sub_C8D290(a2, a2 + 3, v39 + 1, 1);
      v39 = a2[1];
    }
    *(_BYTE *)(*a2 + v39) = 45;
    v23 = a2[1] + 1LL;
    a2[1] = v23;
  }
  else
  {
LABEL_36:
    v23 = a2[1];
  }
  for ( ; v11; a2[1] = v23 )
  {
    if ( (unsigned __int64)(v23 + 1) > a2[2] )
    {
      sub_C8D290(a2, a2 + 3, v23 + 1, 1);
      v23 = a2[1];
    }
    ++v12;
    *(_BYTE *)(*a2 + v23) = v11;
    v11 = *v12;
    v23 = a2[1] + 1LL;
  }
  if ( (_DWORD)v10 != 16 && (_DWORD)v10 != 2 && !v55 )
  {
    v40 = 0;
    v54 = v23;
    while ( 1 )
    {
      v43 = v64;
      if ( v64 > 0x40 )
      {
        if ( v43 == (unsigned int)sub_C444A0((__int64)&v63) )
        {
          v26 = v43;
          result = *a2 + a2[1];
          v44 = (char *)(*a2 + v54);
          if ( (char *)result != v44 )
          {
LABEL_108:
            if ( (unsigned __int64)v44 < --result )
            {
              do
              {
                v45 = *v44;
                v46 = *(_BYTE *)result;
                ++v44;
                --result;
                *(v44 - 1) = v46;
                *(_BYTE *)(result + 1) = v45;
              }
              while ( (unsigned __int64)v44 < result );
              v26 = v64;
            }
            if ( v26 <= 0x40 )
              return result;
          }
LABEL_112:
          if ( v63 )
            return j_j___libc_free_0_0(v63);
          return result;
        }
      }
      else if ( !v63 )
      {
        v26 = v64;
        result = *a2 + a2[1];
        v44 = (char *)(*a2 + v54);
        if ( (char *)result == v44 )
          return result;
        goto LABEL_108;
      }
      sub_C45A90((__int64 **)&v63, (unsigned int)v10, &v63, &v62);
      v41 = a2[1];
      if ( a7 && !(v40 % v61) && v40 )
      {
        if ( (unsigned __int64)(v41 + 1) > a2[2] )
        {
          sub_C8D290(a2, a2 + 3, v41 + 1, 1);
          v41 = a2[1];
        }
        *(_BYTE *)(*a2 + v41) = 39;
        v41 = a2[1] + 1LL;
        a2[1] = v41;
      }
      v42 = *(_BYTE *)(v60 + v62);
      if ( (unsigned __int64)(v41 + 1) > a2[2] )
      {
        sub_C8D290(a2, a2 + 3, v41 + 1, 1);
        v41 = a2[1];
      }
      ++v40;
      *(_BYTE *)(*a2 + v41) = v42;
      ++a2[1];
    }
  }
  v24 = 4;
  if ( (_DWORD)v10 != 16 )
    v24 = 2 * ((_DWORD)v10 == 8) + 1;
  v25 = v10 - 1;
  v51 = v23;
  v26 = v64;
  v27 = 0;
  v53 = v25;
  v48 = a2 + 3;
  v28 = a2;
  v29 = v23;
  v30 = v28;
  while ( v26 > 0x40 )
  {
    v56 = v26;
    v31 = sub_C444A0((__int64)&v63);
    v26 = v56;
    if ( v31 == v56 )
    {
      result = *v30 + v29;
      v44 = (char *)(*v30 + v51);
      if ( v44 == (char *)result )
        goto LABEL_112;
      goto LABEL_108;
    }
    v32 = (unsigned __int64 *)v63;
LABEL_52:
    v33 = *(_DWORD *)v32 & v53;
    if ( a7 && !(v27 % v61) && v27 )
    {
      if ( (unsigned __int64)(v29 + 1) > v30[2] )
      {
        v59 = *(_DWORD *)v32 & v53;
        sub_C8D290(v30, v48, v29 + 1, 1);
        v29 = v30[1];
        v33 = v59;
      }
      *(_BYTE *)(*v30 + v29) = 39;
      v29 = v30[1] + 1LL;
      v30[1] = v29;
    }
    v34 = *(_BYTE *)(v60 + v33);
    if ( (unsigned __int64)(v29 + 1) > v30[2] )
    {
      v57 = v34;
      sub_C8D290(v30, v48, v29 + 1, 1);
      v29 = v30[1];
      v34 = v57;
    }
    *(_BYTE *)(*v30 + v29) = v34;
    v26 = v64;
    v29 = v30[1] + 1LL;
    v30[1] = v29;
    if ( v26 <= 0x40 )
    {
      if ( v24 == v26 )
        v63 = 0;
      else
        v63 >>= v24;
    }
    else
    {
      sub_C482E0((__int64)&v63, v24);
      v29 = v30[1];
      v26 = v64;
    }
    ++v27;
  }
  if ( v63 )
  {
    v32 = &v63;
    goto LABEL_52;
  }
  result = *v30 + v29;
  v44 = (char *)(*v30 + v51);
  if ( v44 != (char *)result )
    goto LABEL_108;
  return result;
}
