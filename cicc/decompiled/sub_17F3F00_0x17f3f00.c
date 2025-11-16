// Function: sub_17F3F00
// Address: 0x17f3f00
//
__int64 __fastcall sub_17F3F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rcx
  _QWORD *v7; // rdx
  _QWORD *v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  _BYTE *v17; // rsi
  __int64 *v18; // rbx
  __int64 *v19; // r12
  unsigned int v20; // r12d
  __int64 v22; // [rsp+8h] [rbp-98h] BYREF
  _QWORD v23[4]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int8 v24; // [rsp+30h] [rbp-70h]
  _BYTE *v25; // [rsp+38h] [rbp-68h] BYREF
  _BYTE *v26; // [rsp+40h] [rbp-60h]
  _BYTE *v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h] BYREF
  __int64 v29; // [rsp+58h] [rbp-48h] BYREF
  _QWORD *v30; // [rsp+60h] [rbp-40h]

  v23[3] = a4;
  v23[0] = a1;
  v4 = (unsigned int)(dword_4FA5BE0 + 2);
  v23[1] = a2;
  v23[2] = a3;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v30 = 0;
  v5 = (_QWORD *)sub_2207820(16 * v4);
  v6 = v5;
  if ( v5 && v4 )
  {
    v7 = &v5[2 * v4];
    do
    {
      *v5 = 0;
      v5 += 2;
      *(v5 - 1) = 0;
    }
    while ( v7 != v5 );
  }
  v8 = v30;
  v30 = v6;
  if ( v8 )
    j_j___libc_free_0_0(v8);
  sub_1695A80((char *)qword_4FA3D40[20], qword_4FA3D40[21], &v28, &v29);
  if ( v25 != v26 )
    v26 = v25;
  v9 = *(_QWORD *)(v23[0] + 80LL);
  v10 = v23[0] + 72LL;
  if ( v23[0] + 72LL != v9 )
  {
    do
    {
      v11 = v9;
      v9 = *(_QWORD *)(v9 + 8);
      v12 = *(_QWORD *)(v11 + 24);
      v13 = v11 + 16;
LABEL_11:
      while ( v13 != v12 )
      {
        while ( 1 )
        {
          v14 = v12;
          v12 = *(_QWORD *)(v12 + 8);
          if ( *(_BYTE *)(v14 - 8) != 78 )
            break;
          v15 = *(_QWORD *)(v14 - 48);
          if ( *(_BYTE *)(v15 + 16) )
            break;
          v16 = *(_DWORD *)(v15 + 36);
          if ( v16 != 135 && v16 != 137 && v16 != 133 )
            break;
          if ( *(_BYTE *)(*(_QWORD *)(v14 - 24 + 24 * (2LL - (*(_DWORD *)(v14 - 4) & 0xFFFFFFF))) + 16LL) == 13 )
            break;
          v22 = v14 - 24;
          v17 = v26;
          if ( v26 == v27 )
          {
            sub_17F26D0((__int64)&v25, v26, &v22);
            goto LABEL_11;
          }
          if ( v26 )
          {
            *(_QWORD *)v26 = v14 - 24;
            v17 = v26;
          }
          v26 = v17 + 8;
          if ( v13 == v12 )
            goto LABEL_22;
        }
      }
LABEL_22:
      ;
    }
    while ( v9 != v10 );
    v18 = (__int64 *)v25;
    v19 = (__int64 *)v26;
    if ( v25 != v26 )
    {
      do
      {
        if ( (unsigned __int8)sub_17F29E0((__int64)v23, *v18) )
          v24 = 1;
        ++v18;
      }
      while ( v19 != v18 );
    }
  }
  v20 = v24;
  if ( v30 )
    j_j___libc_free_0_0(v30);
  if ( v25 )
    j_j___libc_free_0(v25, v27 - v25);
  return v20;
}
