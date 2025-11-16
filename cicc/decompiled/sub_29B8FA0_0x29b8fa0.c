// Function: sub_29B8FA0
// Address: 0x29b8fa0
//
void __fastcall sub_29B8FA0(unsigned __int64 *a1, __int64 *a2, __int64 *a3, _QWORD *a4)
{
  _QWORD *v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  bool v15; // cf
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdi
  _QWORD *v22; // r13
  _QWORD *v23; // rbx
  __int64 v24; // rdx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r13
  __int64 v28; // rax
  unsigned __int64 v29; // r13
  _QWORD *v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  unsigned __int64 v32; // [rsp+20h] [rbp-40h]
  _QWORD *v33; // [rsp+28h] [rbp-38h]

  v7 = (_QWORD *)a1[1];
  if ( v7 != (_QWORD *)a1[2] )
  {
    if ( v7 )
    {
      v8 = *a2;
      v9 = *a3;
      v10 = *a4;
      v7[1] = 0;
      *v7 = v8;
      v7[2] = v9;
      v7[3] = v10;
      v7[4] = 0;
      v7[5] = 0;
      v7[6] = 0;
      v7[7] = 0;
      v7[8] = 0;
      v7[9] = 0;
      v7[10] = 0;
      v7[11] = 0;
      v7[12] = 0;
      v7[13] = 0;
      v7 = (_QWORD *)a1[1];
    }
    a1[1] = (unsigned __int64)(v7 + 14);
    return;
  }
  v11 = *a1;
  v12 = (__int64)v7 - *a1;
  v13 = 0x6DB6DB6DB6DB6DB7LL * (v12 >> 4);
  if ( v13 == 0x124924924924924LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v14 = 1;
  if ( v13 )
    v14 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)((__int64)v7 - v11) >> 4);
  v15 = __CFADD__(v14, v13);
  v16 = v14 + v13;
  if ( v15 )
  {
    v27 = 0x7FFFFFFFFFFFFFC0LL;
LABEL_30:
    v30 = a4;
    v28 = sub_22077B0(v27);
    v12 = (__int64)v7 - v11;
    v29 = v28 + v27;
    v33 = (_QWORD *)v28;
    a4 = v30;
    v17 = v28 + 112;
    v32 = v29;
    goto LABEL_11;
  }
  if ( v16 )
  {
    if ( v16 > 0x124924924924924LL )
      v16 = 0x124924924924924LL;
    v27 = 112 * v16;
    goto LABEL_30;
  }
  v32 = 0;
  v17 = 112;
  v33 = 0;
LABEL_11:
  v18 = (_QWORD *)((char *)v33 + v12);
  if ( v18 )
  {
    v19 = *a3;
    v20 = *a4;
    v18[1] = 0;
    v21 = *a2;
    v18[4] = 0;
    v18[2] = v19;
    *v18 = v21;
    v18[3] = v20;
    v18[5] = 0;
    v18[6] = 0;
    v18[7] = 0;
    v18[8] = 0;
    v18[9] = 0;
    v18[10] = 0;
    v18[11] = 0;
    v18[12] = 0;
    v18[13] = 0;
  }
  if ( v7 != (_QWORD *)v11 )
  {
    v22 = v33;
    v23 = (_QWORD *)v11;
    while ( 1 )
    {
      if ( v22 )
      {
        *v22 = *v23;
        v22[1] = v23[1];
        v22[2] = v23[2];
        v22[3] = v23[3];
        v22[4] = v23[4];
        v22[5] = v23[5];
        v22[6] = v23[6];
        v22[7] = v23[7];
        v22[8] = v23[8];
        v22[9] = v23[9];
        v22[10] = v23[10];
        v24 = v23[11];
        v23[10] = 0;
        v23[9] = 0;
        v23[8] = 0;
        v22[11] = v24;
        v22[12] = v23[12];
        v22[13] = v23[13];
        v23[13] = 0;
        v23[11] = 0;
      }
      else
      {
        v26 = v23[11];
        if ( v26 )
          j_j___libc_free_0(v26);
      }
      v25 = v23[8];
      if ( v25 )
        j_j___libc_free_0(v25);
      v23 += 14;
      if ( v7 == v23 )
        break;
      v22 += 14;
    }
    v17 = (__int64)(v22 + 28);
  }
  if ( v11 )
  {
    v31 = v17;
    j_j___libc_free_0(v11);
    v17 = v31;
  }
  a1[1] = v17;
  *a1 = (unsigned __int64)v33;
  a1[2] = v32;
}
