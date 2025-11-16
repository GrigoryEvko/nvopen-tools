// Function: sub_390F620
// Address: 0x390f620
//
bool __fastcall sub_390F620(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // r9d
  __int64 v11; // r12
  unsigned __int64 v12; // rbx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r15
  __int64 v16; // rdi
  __int64 *v17; // r13
  double v18; // xmm1_8
  double v19; // xmm1_8
  __int64 v20; // rbx
  double v21; // xmm3_8
  double v22; // xmm0_8
  __int64 *v23; // rax
  __int64 v24; // [rsp+0h] [rbp-80h]
  double v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  unsigned __int64 v29; // [rsp+30h] [rbp-50h]
  double v31; // [rsp+40h] [rbp-40h]

  if ( !*(_BYTE *)(a2 + 56) )
    return 0;
  v24 = *(_QWORD *)(a2 + 64);
  v29 = sub_390F280(a1, a2, (__int64)a3, a4, a5, a6);
  if ( !v29 )
    return 0;
  v28 = *(unsigned int *)(*(_QWORD *)(a2 + 24) + 24LL);
  v27 = 0;
  v26 = 0;
  v11 = sub_390EBE0(a1, a2, (__int64)a3, v8, v9, v10);
  v25 = 1.797693134862316e308;
  do
  {
    v12 = 0;
    *(_QWORD *)(a2 + 64) = v27;
    sub_38CFC60((__int64)a3, (_QWORD *)a2);
    v31 = 0.0;
    do
    {
      v13 = *(__int64 **)(a1 + 32);
      if ( v13 == *(__int64 **)(a1 + 24) )
        v14 = *(unsigned int *)(a1 + 44);
      else
        v14 = *(unsigned int *)(a1 + 40);
      v15 = &v13[v14];
      if ( v13 == v15 )
      {
LABEL_11:
        v18 = 0.0;
      }
      else
      {
        while ( 1 )
        {
          v16 = *v13;
          v17 = v13;
          if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v15 == ++v13 )
            goto LABEL_11;
        }
        v18 = 0.0;
        if ( v13 != v15 )
        {
          do
          {
            v22 = sub_390E4A0(v16, v11, v12, a3);
            v23 = v17 + 1;
            v18 = v18 + v22;
            if ( v17 + 1 == v15 )
              break;
            while ( 1 )
            {
              v16 = *v23;
              v17 = v23;
              if ( (unsigned __int64)*v23 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v15 == ++v23 )
                goto LABEL_12;
            }
          }
          while ( v23 != v15 );
        }
      }
LABEL_12:
      v19 = fmax(v18, v31);
      v12 += v28;
      v31 = v19;
    }
    while ( v29 > v12 );
    v20 = v26;
    v21 = fmin(v19, v25);
    if ( v25 > v19 )
      v20 = v27;
    v26 = v20;
    v25 = v21;
    if ( v21 == 0.0 )
      break;
    ++v27;
  }
  while ( v29 != v27 );
  *(_QWORD *)(a2 + 64) = v20;
  sub_38CFC60((__int64)a3, (_QWORD *)a2);
  return v20 != v24;
}
