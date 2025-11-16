// Function: sub_FE98F0
// Address: 0xfe98f0
//
__int64 __fastcall sub_FE98F0(_QWORD *a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  unsigned __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rbx
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  __int16 v16; // ax
  __int64 v17; // rax
  __int16 v18; // r12
  int v19; // r11d
  int v20; // r11d
  _QWORD *v21; // rax
  __int64 *v22; // rdx
  _QWORD *v23; // r14
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  _QWORD *v29; // rbx
  _QWORD *v30; // rdi
  __int64 result; // rax
  __int64 *v32; // rax
  __int16 v33; // [rsp+1Eh] [rbp-72h]
  unsigned __int64 v34; // [rsp+20h] [rbp-70h] BYREF
  __int16 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  __int16 v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v39; // [rsp+48h] [rbp-48h]
  __int64 v40; // [rsp+50h] [rbp-40h]
  __int64 *v41; // [rsp+58h] [rbp-38h]

  v1 = 0;
  v3 = -1;
  v34 = 0;
  v4 = a1[8];
  v35 = 0;
  v33 = 0x3FFF;
  if ( a1[9] != v4 )
  {
    do
    {
      v6 = 24 * v1 + a1[1];
      if ( (int)sub_D788E0(*(_QWORD *)v6, *(_WORD *)(v6 + 8), v3, v33) < 0 )
      {
        v3 = *(_QWORD *)v6;
        v33 = *(_WORD *)(v6 + 8);
      }
      v5 = a1[1] + 24 * v1;
      if ( (int)sub_D788E0(v34, v35, *(_QWORD *)v5, *(_WORD *)(v5 + 8)) < 0 )
      {
        v34 = *(_QWORD *)v5;
        v35 = *(_WORD *)(v5 + 8);
      }
      ++v1;
    }
    while ( v1 < 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[9] - a1[8]) >> 3) );
  }
  v38 = 1;
  v7 = 0;
  LOWORD(v39) = 54;
  v8 = sub_FDE760((__int64)&v38, (__int64)&v34);
  v9 = a1[1];
  v10 = *(_QWORD *)v8;
  v11 = v9;
  v37 = *(_WORD *)(v8 + 8);
  v36 = v10;
  if ( a1[2] != v9 )
  {
    do
    {
      v14 = 24 * v7;
      v15 = 24 * v7 + v9;
      v16 = *(_WORD *)(v15 + 8);
      v38 = *(_QWORD *)v15;
      LOWORD(v39) = v16;
      v17 = sub_FE9650((__int64)&v38, (__int64)&v36);
      v18 = *(_WORD *)(v17 + 8);
      v12 = *(_QWORD *)v17;
      v19 = sub_D788E0(*(_QWORD *)v17, v18, 1u, 0);
      v13 = 1;
      if ( v19 >= 0 )
      {
        v20 = sub_D788E0(v12, v18, 0xFFFFFFFFFFFFFFFFLL, 0);
        v13 = -1;
        if ( v20 < 0 )
        {
          if ( v18 <= 0 )
          {
            if ( v18 )
              v12 >>= -(char)v18;
          }
          else
          {
            v12 <<= v18;
          }
          v13 = 1;
          if ( v12 )
            v13 = v12;
        }
      }
      ++v7;
      *(_QWORD *)(a1[1] + v14 + 16) = v13;
      v11 = a1[2];
      v9 = a1[1];
    }
    while ( 0xAAAAAAAAAAAAAAABLL * ((v11 - v9) >> 3) > v7 );
  }
  v21 = (_QWORD *)a1[4];
  v22 = (__int64 *)a1[5];
  v23 = a1 + 4;
  a1[2] = 0;
  v24 = a1[6];
  v25 = a1[3];
  a1[1] = 0;
  a1[3] = 0;
  v38 = (__int64)v21;
  v39 = v22;
  v40 = v24;
  if ( a1 + 4 == v21 )
  {
    v26 = &v38;
    v39 = &v38;
    v38 = (__int64)&v38;
  }
  else
  {
    *v22 = (__int64)&v38;
    v26 = (__int64 *)v38;
    *(_QWORD *)(v38 + 8) = &v38;
    a1[5] = v23;
    a1[4] = v23;
    a1[6] = 0;
  }
  v41 = v26;
  sub_FE9370((__int64)a1);
  v27 = a1[1];
  v28 = a1[3];
  a1[1] = v9;
  a1[2] = v11;
  a1[3] = v25;
  if ( v27 )
    j_j___libc_free_0(v27, v28 - v27);
  v29 = (_QWORD *)a1[4];
  while ( v23 != v29 )
  {
    v30 = v29;
    v29 = (_QWORD *)*v29;
    j_j___libc_free_0(v30, 40);
  }
  result = v38;
  if ( (__int64 *)v38 == &v38 )
  {
    a1[5] = v23;
    a1[4] = v23;
    a1[6] = 0;
  }
  else
  {
    a1[4] = v38;
    v32 = v39;
    a1[5] = v39;
    *v32 = (__int64)v23;
    *(_QWORD *)(a1[4] + 8LL) = v23;
    result = v40;
    v23 = (_QWORD *)a1[4];
    a1[6] = v40;
  }
  a1[7] = v23;
  return result;
}
