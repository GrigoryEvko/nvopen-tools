// Function: sub_35BEEC0
// Address: 0x35beec0
//
unsigned __int64 __fastcall sub_35BEEC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 result; // rax
  unsigned int v5; // r15d
  _DWORD *v6; // rsi
  unsigned int v7; // r8d
  unsigned int v8; // r11d
  unsigned int v9; // r13d
  unsigned int v10; // r12d
  __int64 v11; // r15
  unsigned __int64 i; // rdx
  unsigned int **v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // r14d
  unsigned int v16; // ecx
  __int64 v17; // r8
  __int64 v18; // rsi
  float v19; // xmm0_4
  bool v20; // zf
  unsigned __int64 v21; // r14
  float *v22; // rax
  float *v23; // rcx
  volatile signed __int32 *v24; // rdi
  volatile signed __int32 *v25; // r14
  __int64 v26; // rsi
  unsigned int v27; // ecx
  __int64 v28; // rcx
  __int64 v29; // rsi
  unsigned __int64 v30; // r10
  __int64 v31; // r9
  __int64 *v32; // rdx
  __int64 *v33; // rsi
  __int64 v34; // [rsp+20h] [rbp-90h]
  int v35; // [rsp+2Ch] [rbp-84h]
  const void **v36; // [rsp+30h] [rbp-80h]
  float v37; // [rsp+38h] [rbp-78h]
  unsigned __int64 v38; // [rsp+38h] [rbp-78h]
  __int64 v39; // [rsp+40h] [rbp-70h]
  unsigned int v40; // [rsp+48h] [rbp-68h]
  __int64 *v41; // [rsp+48h] [rbp-68h]
  __int64 v42; // [rsp+50h] [rbp-60h]
  unsigned __int64 v43; // [rsp+50h] [rbp-60h]
  __int64 v44; // [rsp+58h] [rbp-58h]
  __int64 v45; // [rsp+60h] [rbp-50h] BYREF
  volatile signed __int32 *v46; // [rsp+68h] [rbp-48h]
  unsigned int v47; // [rsp+70h] [rbp-40h] BYREF
  float *v48; // [rsp+78h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  v47 = 0;
  v44 = v2;
  result = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a2 + 168) - *(_QWORD *)(a2 + 160)) >> 5);
  if ( (_DWORD)result )
  {
    do
    {
      v6 = *(_DWORD **)(a2 + 192);
      if ( v6 == sub_35B8320(*(_DWORD **)(a2 + 184), (__int64)v6, (int *)&v47) )
      {
        result = *(_QWORD *)(a2 + 160);
        v5 = v47;
        v35 = -1431655765 * ((__int64)(*(_QWORD *)(a2 + 168) - result) >> 5);
        goto LABEL_6;
      }
      v5 = v47 + 1;
      v47 = v5;
    }
    while ( v7 > v5 );
    result = *(_QWORD *)(a2 + 160);
    v35 = -1431655765 * ((__int64)(*(_QWORD *)(a2 + 168) - result) >> 5);
LABEL_6:
    if ( v35 != v5 )
    {
      v9 = v5;
      v10 = v8;
      v11 = a2;
      v34 = a2 + 88;
      for ( i = result; ; i = *(_QWORD *)(v11 + 160) )
      {
        v13 = (unsigned int **)(96LL * v9 + i);
        v42 = 96LL * v9;
        v14 = *(unsigned int *)(v44 + 160);
        v15 = *((_DWORD *)v13 + 10);
        v16 = v15 & 0x7FFFFFFF;
        v17 = 8LL * (v15 & 0x7FFFFFFF);
        if ( (v15 & 0x7FFFFFFFu) < (unsigned int)v14 && (v18 = *(_QWORD *)(*(_QWORD *)(v44 + 152) + 8LL * v16)) != 0 )
        {
          v19 = *(float *)(v18 + 116);
          v20 = v19 == 0.0;
        }
        else
        {
          v27 = v16 + 1;
          if ( (unsigned int)v14 >= v27 || v27 == v14 )
          {
            v28 = *(_QWORD *)(v44 + 152);
          }
          else if ( v27 >= v14 )
          {
            v30 = v27 - v14;
            v31 = *(_QWORD *)(v44 + 168);
            if ( v27 > (unsigned __int64)*(unsigned int *)(v44 + 164) )
            {
              v38 = v27 - v14;
              v39 = *(_QWORD *)(v44 + 168);
              sub_C8D5F0(v44 + 152, (const void *)(v44 + 168), v27, 8u, v17, v31);
              v30 = v38;
              v31 = v39;
              v17 = 8LL * (v15 & 0x7FFFFFFF);
              v14 = *(unsigned int *)(v44 + 160);
            }
            v28 = *(_QWORD *)(v44 + 152);
            v32 = (__int64 *)(v28 + 8 * v14);
            v33 = &v32[v30];
            if ( v32 != v33 )
            {
              do
                *v32++ = v31;
              while ( v33 != v32 );
              LODWORD(v14) = *(_DWORD *)(v44 + 160);
              v28 = *(_QWORD *)(v44 + 152);
            }
            *(_DWORD *)(v44 + 160) = v30 + v14;
          }
          else
          {
            *(_DWORD *)(v44 + 160) = v27;
            v28 = *(_QWORD *)(v44 + 152);
          }
          v41 = (__int64 *)(v28 + v17);
          v29 = sub_2E10F30(v15);
          *v41 = v29;
          sub_2E11E80((_QWORD *)v44, v29);
          v13 = (unsigned int **)(*(_QWORD *)(v11 + 160) + v42);
          v19 = *(float *)(v29 + 116);
          v20 = v19 == 0.0;
        }
        if ( v20 )
          v37 = 1.1754944e-38;
        else
          v37 = v19 + 10.0;
        v36 = (const void **)*v13;
        v21 = 4LL * **v13;
        v40 = **v13;
        v22 = (float *)sub_2207820(v21);
        v23 = v22;
        if ( v22 && v40 )
          v23 = (float *)memset(v22, 0, v21);
        if ( v21 )
          v23 = (float *)memmove(v23, v36[1], v21);
        v48 = v23;
        *v23 = v37;
        v47 = v40;
        sub_35BE780(&v45, v34, &v47);
        if ( v48 )
          j_j___libc_free_0_0((unsigned __int64)v48);
        result = *(_QWORD *)(v11 + 160) + v42;
        *(_QWORD *)result = v45;
        v24 = *(volatile signed __int32 **)(result + 8);
        v25 = v46;
        if ( v46 != v24 )
        {
          if ( v46 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd(v46 + 2, 1u);
            else
              ++*((_DWORD *)v46 + 2);
            v24 = *(volatile signed __int32 **)(result + 8);
          }
          if ( v24 )
          {
            v43 = result;
            sub_A191D0(v24);
            result = v43;
          }
          *(_QWORD *)(result + 8) = v25;
          v24 = v46;
        }
        if ( v24 )
          result = sub_A191D0(v24);
        v47 = ++v9;
        if ( v10 > v9 )
        {
          while ( 1 )
          {
            v26 = *(_QWORD *)(v11 + 192);
            result = (unsigned __int64)sub_35B8320(*(_DWORD **)(v11 + 184), v26, (int *)&v47);
            if ( v26 == result )
              break;
            result = v47;
            v9 = v47 + 1;
            v47 = v9;
            if ( v10 <= v9 )
              goto LABEL_35;
          }
          v9 = v47;
        }
LABEL_35:
        if ( v35 == v9 )
          break;
      }
    }
  }
  return result;
}
