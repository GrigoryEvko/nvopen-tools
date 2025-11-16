// Function: sub_2B14D60
// Address: 0x2b14d60
//
_QWORD *__fastcall sub_2B14D60(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *result; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+10h] [rbp-30h]

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = a1;
  v4 = (a2 - (__int64)a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_35:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          return (_QWORD *)a2;
        goto LABEL_55;
      }
      v15 = *v3;
      v21 = 6;
      v22 = 0;
      v23 = *(_QWORD *)(v15 + 96);
      v16 = v23;
      if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
      {
        sub_BD6050(&v21, *(_QWORD *)(v15 + 80) & 0xFFFFFFFFFFFFFFF8LL);
        v16 = v23;
      }
      result = v3;
      if ( !v16 )
        return result;
      if ( v16 != -4096 && v16 != -8192 )
        sub_BD60C0(&v21);
      ++v3;
    }
    v17 = *v3;
    v21 = 6;
    v22 = 0;
    v23 = *(_QWORD *)(v17 + 96);
    v18 = v23;
    if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
    {
      sub_BD6050(&v21, *(_QWORD *)(v17 + 80) & 0xFFFFFFFFFFFFFFF8LL);
      v18 = v23;
    }
    result = v3;
    if ( !v18 )
      return result;
    if ( v18 != -4096 && v18 != -8192 )
      sub_BD60C0(&v21);
    ++v3;
LABEL_55:
    v19 = *v3;
    v21 = 6;
    v22 = 0;
    v23 = *(_QWORD *)(v19 + 96);
    v20 = v23;
    if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
    {
      sub_BD6050(&v21, *(_QWORD *)(v19 + 80) & 0xFFFFFFFFFFFFFFF8LL);
      v20 = v23;
    }
    result = v3;
    if ( !v20 )
      return result;
    if ( v20 != -8192 && v20 != -4096 )
      sub_BD60C0(&v21);
    return (_QWORD *)a2;
  }
  v5 = &a1[4 * v2];
  while ( 1 )
  {
    v12 = *v3;
    v21 = 6;
    v22 = 0;
    v23 = *(_QWORD *)(v12 + 96);
    v13 = v23;
    if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
    {
      sub_BD6050(&v21, *(_QWORD *)(v12 + 80) & 0xFFFFFFFFFFFFFFF8LL);
      v13 = v23;
    }
    if ( !v13 )
      return v3;
    if ( v13 != -4096 && v13 != -8192 )
      sub_BD60C0(&v21);
    v6 = v3[1];
    v21 = 6;
    v22 = 0;
    v23 = *(_QWORD *)(v6 + 96);
    v7 = v23;
    if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
    {
      sub_BD6050(&v21, *(_QWORD *)(v6 + 80) & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v23;
    }
    if ( !v7 )
      return v3 + 1;
    if ( v7 != -4096 && v7 != -8192 )
      sub_BD60C0(&v21);
    v8 = v3[2];
    v21 = 6;
    v22 = 0;
    v23 = *(_QWORD *)(v8 + 96);
    v9 = v23;
    if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
    {
      sub_BD6050(&v21, *(_QWORD *)(v8 + 80) & 0xFFFFFFFFFFFFFFF8LL);
      v9 = v23;
    }
    if ( !v9 )
      return v3 + 2;
    if ( v9 != -4096 && v9 != -8192 )
      sub_BD60C0(&v21);
    v10 = v3[3];
    v21 = 6;
    v22 = 0;
    v23 = *(_QWORD *)(v10 + 96);
    v11 = v23;
    if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
    {
      sub_BD6050(&v21, *(_QWORD *)(v10 + 80) & 0xFFFFFFFFFFFFFFF8LL);
      v11 = v23;
    }
    if ( !v11 )
      return v3 + 3;
    if ( v11 != -4096 && v11 != -8192 )
      sub_BD60C0(&v21);
    v3 += 4;
    if ( v5 == v3 )
    {
      v4 = (a2 - (__int64)v3) >> 3;
      goto LABEL_35;
    }
  }
}
