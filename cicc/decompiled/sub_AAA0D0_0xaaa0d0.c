// Function: sub_AAA0D0
// Address: 0xaaa0d0
//
__int64 __fastcall sub_AAA0D0(unsigned __int8 *a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // bl
  __int64 v3; // r15
  unsigned __int8 *v4; // r13
  int v5; // edx
  __int64 result; // rax
  unsigned int v7; // r14d
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int16 v10; // ax
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 i; // r14
  __int64 v15; // r10
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned int v18; // ebx
  unsigned __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rbx
  unsigned int v23; // eax
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rdx
  int v27; // ebx
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-C8h]
  __int64 v30; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v31; // [rsp+18h] [rbp-B8h]
  __int64 v32; // [rsp+18h] [rbp-B8h]
  __int64 v33; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-A8h]
  __int64 v35; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-98h]
  char v37; // [rsp+3Ch] [rbp-94h]
  __int64 v38; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v39; // [rsp+48h] [rbp-88h]
  unsigned __int8 *v40; // [rsp+50h] [rbp-80h] BYREF
  __int64 v41; // [rsp+58h] [rbp-78h]
  _BYTE v42[112]; // [rsp+60h] [rbp-70h] BYREF

  v2 = *a1;
  v3 = *((_QWORD *)a1 + 1);
  if ( *a1 == 13 )
    return sub_ACADE0(*(_QWORD *)(v3 + 24));
  v4 = a2;
  v5 = *a2;
  if ( (unsigned int)(v5 - 12) <= 1 )
    return sub_ACADE0(*(_QWORD *)(v3 + 24));
  if ( (unsigned int)v2 - 12 <= 1 )
    return sub_ACA8A0(*(_QWORD *)(v3 + 24));
  if ( (_BYTE)v5 != 17 )
    return 0;
  if ( *(_BYTE *)(v3 + 8) == 17 )
  {
    v7 = *((_DWORD *)a2 + 8);
    v8 = *(unsigned int *)(v3 + 32);
    if ( v7 > 0x40 )
    {
      v31 = *(unsigned int *)(v3 + 32);
      v20 = sub_C444A0(a2 + 24);
      v8 = v31;
      if ( v7 - v20 > 0x40 )
        return sub_ACADE0(*(_QWORD *)(v3 + 24));
      v9 = **((_QWORD **)a2 + 3);
    }
    else
    {
      v9 = *((_QWORD *)a2 + 3);
    }
    if ( v8 > v9 )
      goto LABEL_11;
    return sub_ACADE0(*(_QWORD *)(v3 + 24));
  }
LABEL_11:
  if ( v2 != 5 )
  {
LABEL_29:
    result = sub_AD6D20(a1, a2);
    if ( result )
      return result;
    v18 = *((_DWORD *)a2 + 8);
    if ( v18 <= 0x40 )
    {
      v19 = *((_QWORD *)a2 + 3);
      goto LABEL_32;
    }
    if ( v18 - (unsigned int)sub_C444A0(a2 + 24) <= 0x40 )
    {
      v19 = **((_QWORD **)a2 + 3);
LABEL_32:
      if ( *(unsigned int *)(v3 + 32) > v19 )
        return sub_AD7630(a1, 0);
    }
    return 0;
  }
  v10 = *((_WORD *)a1 + 1);
  if ( v10 != 34 )
  {
    if ( v10 == 62 )
    {
      v22 = *(_QWORD *)&a1[32 * (2LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
      if ( *(_BYTE *)v22 == 17 )
      {
        v23 = *((_DWORD *)a2 + 8);
        v39 = v23;
        if ( v23 > 0x40 )
        {
          sub_C43780(&v38, a2 + 24);
          v23 = v39;
          v24 = v38;
        }
        else
        {
          v24 = *((_QWORD *)a2 + 3);
          v38 = v24;
        }
        LODWORD(v41) = v23;
        v40 = (unsigned __int8 *)v24;
        BYTE4(v41) = 1;
        v25 = *(_DWORD *)(v22 + 32);
        v39 = 0;
        v34 = v25;
        if ( v25 > 0x40 )
        {
          sub_C43780(&v33, v22 + 24);
          v25 = v34;
          v26 = v33;
        }
        else
        {
          v26 = *(_QWORD *)(v22 + 24);
          v33 = v26;
        }
        v36 = v25;
        v35 = v26;
        v34 = 0;
        v37 = 1;
        v27 = sub_AA8A40(&v35, (__int64 *)&v40);
        if ( v36 > 0x40 && v35 )
          j_j___libc_free_0_0(v35);
        if ( v34 > 0x40 && v33 )
          j_j___libc_free_0_0(v33);
        if ( (unsigned int)v41 > 0x40 && v40 )
          j_j___libc_free_0_0(v40);
        if ( v39 > 0x40 && v38 )
          j_j___libc_free_0_0(v38);
        v28 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
        if ( v27 )
          return sub_AD5840(*(_QWORD *)&a1[-32 * v28], a2, 0);
        else
          return *(_QWORD *)&a1[32 * (1 - v28)];
      }
    }
    goto LABEL_29;
  }
  v40 = v42;
  v41 = 0x800000000LL;
  v11 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
  v12 = v11;
  if ( v11 > 8 )
  {
    a2 = v42;
    sub_C8D5F0(&v40, v42, v11, 8);
    v11 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
  }
  if ( (_DWORD)v11 )
  {
    v13 = (unsigned int)(v11 - 1);
    for ( i = 0; ; ++i )
    {
      v15 = *(_QWORD *)&a1[32 * (i - v11)];
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + 8LL) - 17 <= 1 )
      {
        a2 = v4;
        result = sub_AD5840(v15, v4, 0);
        if ( !result )
          goto LABEL_37;
        v12 = (unsigned int)v41;
        if ( (unsigned __int64)(unsigned int)v41 + 1 > HIDWORD(v41) )
        {
          a2 = v42;
          v29 = result;
          sub_C8D5F0(&v40, v42, (unsigned int)v41 + 1LL, 8);
          v12 = (unsigned int)v41;
          result = v29;
        }
        *(_QWORD *)&v40[8 * v12] = result;
        LODWORD(v41) = v41 + 1;
      }
      else
      {
        v16 = (unsigned int)v41;
        v17 = (unsigned int)v41 + 1LL;
        if ( v17 > HIDWORD(v41) )
        {
          a2 = v42;
          v30 = v15;
          sub_C8D5F0(&v40, v42, v17, 8);
          v16 = (unsigned int)v41;
          v15 = v30;
        }
        v12 = (__int64)v40;
        *(_QWORD *)&v40[8 * v16] = v15;
        LODWORD(v41) = v41 + 1;
      }
      if ( i == v13 )
        break;
      v11 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
    }
  }
  v21 = sub_BB5290(a1, a2, v12);
  a2 = v40;
  result = sub_ADABF0(a1, v40, (unsigned int)v41, *(_QWORD *)(v3 + 24), 0, v21);
LABEL_37:
  if ( v40 != v42 )
  {
    v32 = result;
    _libc_free(v40, a2);
    return v32;
  }
  return result;
}
