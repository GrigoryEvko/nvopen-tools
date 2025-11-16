// Function: sub_FE1050
// Address: 0xfe1050
//
__int64 __fastcall sub_FE1050(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 *v9; // r14
  __int64 result; // rax
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r14
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r11
  int v18; // eax
  unsigned __int64 *v19; // rax
  __int64 *v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+18h] [rbp-98h]
  __int64 v23; // [rsp+30h] [rbp-80h]
  unsigned int v24; // [rsp+30h] [rbp-80h]
  __int64 v25; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-68h]
  __int64 v27; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+58h] [rbp-58h]
  _QWORD *v29; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+68h] [rbp-48h]
  _QWORD *v31; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+78h] [rbp-38h]

  v26 = 128;
  sub_C43690((__int64)&v25, a3, 0);
  v5 = *a1;
  LODWORD(v31) = sub_FDD0F0(*a1, a2);
  v6 = sub_FE8720(v5, &v31);
  v28 = 128;
  sub_C43690((__int64)&v27, v6, 0);
  v30 = 128;
  sub_C43690((__int64)&v29, 0, 0);
  v7 = *(unsigned __int64 **)(a4 + 8);
  if ( *(_BYTE *)(a4 + 28) )
    v8 = *(unsigned int *)(a4 + 20);
  else
    v8 = *(unsigned int *)(a4 + 16);
  v9 = &v7[v8];
  if ( v7 != v9 )
  {
    while ( *v7 >= 0xFFFFFFFFFFFFFFFELL )
    {
      if ( v9 == ++v7 )
        goto LABEL_6;
    }
    if ( v9 != v7 )
    {
      v11 = v9;
      v12 = v7;
      v13 = *v7;
      do
      {
        v23 = *a1;
        LODWORD(v31) = sub_FDD0F0(*a1, v13);
        v14 = sub_FE8720(v23, &v31);
        if ( v30 > 0x40 )
        {
          *v29 = v14;
          memset(v29 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v30 + 63) >> 6) - 8);
        }
        else
        {
          v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v30) & v14;
          if ( !v30 )
            v15 = 0;
          v29 = (_QWORD *)v15;
        }
        sub_C47360((__int64)&v29, &v25);
        sub_C4A1D0((__int64)&v31, (__int64)&v29, (__int64)&v27);
        if ( v30 > 0x40 && v29 )
          j_j___libc_free_0_0(v29);
        v16 = (__int64)v31;
        v29 = v31;
        v30 = v32;
        v17 = *a1;
        v24 = v32;
        if ( v32 > 0x40 )
        {
          v21 = v31;
          v22 = *a1;
          v18 = sub_C444A0((__int64)&v29);
          v17 = v22;
          v16 = -1;
          if ( v24 - v18 <= 0x40 )
            v16 = *v21;
        }
        sub_FE0A90(v17, v13, v16);
        v19 = v12 + 1;
        if ( v12 + 1 == v11 )
          break;
        v13 = *v19;
        for ( ++v12; *v19 >= 0xFFFFFFFFFFFFFFFELL; v12 = v19 )
        {
          if ( v11 == ++v19 )
            goto LABEL_6;
          v13 = *v19;
        }
      }
      while ( v11 != v12 );
    }
  }
LABEL_6:
  result = sub_FE0A90(*a1, a2, a3);
  if ( v30 > 0x40 && v29 )
    result = j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    result = j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 )
  {
    if ( v25 )
      return j_j___libc_free_0_0(v25);
  }
  return result;
}
