// Function: sub_A28440
// Address: 0xa28440
//
__int64 __fastcall sub_A28440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rbx
  __int64 *v13; // r12
  __int64 v14; // r10
  __int64 v15; // r14
  __int64 v16; // r13
  __int64 v17; // r12
  _QWORD *v18; // rbx
  _QWORD *v19; // r14
  unsigned __int64 *v20; // rdx
  unsigned __int64 v21; // rdi
  __int64 v22; // r10
  int v23; // r15d
  _QWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  _QWORD *v27; // r14
  __int64 v28; // rax
  _QWORD *v29; // r13
  __int64 v30; // r12
  unsigned __int64 *v31; // rdx
  unsigned __int64 v32; // rsi
  __int64 v33; // r10
  int v34; // ebx
  _QWORD *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // [rsp+0h] [rbp-70h]
  __int64 i; // [rsp+8h] [rbp-68h]
  __int64 *v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+18h] [rbp-58h]
  __int64 *v42; // [rsp+18h] [rbp-58h]
  __int64 *v43; // [rsp+20h] [rbp-50h]
  unsigned __int64 v44; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 *v45[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1;
  v7 = a1 + 24;
  v8 = v6 + 600;
  *(_QWORD *)(v7 - 16) = a3;
  *(_QWORD *)(v7 - 24) = a4;
  *(_QWORD *)(v7 - 8) = a2;
  sub_A47960(v7, a2, a5);
  v10 = *(_QWORD *)(v6 + 144) - *(_QWORD *)(v6 + 136);
  *(_QWORD *)(v6 + 584) = a6;
  result = v10 >> 4;
  *(_QWORD *)(v6 + 616) = v6 + 600;
  *(_DWORD *)(v6 + 600) = 0;
  *(_QWORD *)(v6 + 608) = 0;
  *(_QWORD *)(v6 + 624) = v6 + 600;
  *(_QWORD *)(v6 + 632) = 0;
  *(_QWORD *)(v6 + 648) = 0;
  *(_DWORD *)(v6 + 640) = result;
  if ( a6 )
  {
    result = *(_QWORD *)(a6 + 24);
    v38 = a6 + 8;
    for ( i = result; v38 != result; i = result )
    {
      v12 = v8;
      v13 = *(__int64 **)(i + 56);
      v43 = *(__int64 **)(i + 64);
      if ( v13 != v43 )
      {
        do
        {
          v14 = *v13;
          if ( *(_DWORD *)(*v13 + 8) == 1 )
          {
            v15 = *(_QWORD *)(v14 + 64);
            if ( v15 + 16LL * *(unsigned int *)(v14 + 72) != v15 )
            {
              v40 = v13;
              v16 = v6;
              v17 = v12;
              v18 = *(_QWORD **)(v14 + 64);
              v41 = v14;
              v19 = (_QWORD *)(v15 + 16LL * *(unsigned int *)(v14 + 72));
              do
              {
                while ( 1 )
                {
                  v20 = (unsigned __int64 *)(*v18 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (*v18 & 1) == 0 || !v20[1] )
                    break;
                  v18 += 2;
                  if ( v19 == v18 )
                    goto LABEL_19;
                }
                v21 = *v20;
                v22 = v17;
                v23 = *(_DWORD *)(v16 + 640) + 1;
                v24 = *(_QWORD **)(v16 + 608);
                v44 = *v20;
                *(_DWORD *)(v16 + 640) = v23;
                if ( !v24 )
                  goto LABEL_17;
                do
                {
                  while ( 1 )
                  {
                    v25 = v24[2];
                    v26 = v24[3];
                    if ( v21 <= v24[4] )
                      break;
                    v24 = (_QWORD *)v24[3];
                    if ( !v26 )
                      goto LABEL_15;
                  }
                  v22 = (__int64)v24;
                  v24 = (_QWORD *)v24[2];
                }
                while ( v25 );
LABEL_15:
                if ( v17 == v22 || v21 < *(_QWORD *)(v22 + 32) )
                {
LABEL_17:
                  v45[0] = &v44;
                  v22 = sub_A28390((_QWORD *)(v16 + 592), (_QWORD *)v22, v45);
                }
                v18 += 2;
                *(_DWORD *)(v22 + 40) = v23;
              }
              while ( v19 != v18 );
LABEL_19:
              v12 = v17;
              v14 = v41;
              v13 = v40;
              v6 = v16;
            }
            v27 = *(_QWORD **)(v14 + 40);
            v28 = *(unsigned int *)(v14 + 48);
            if ( &v27[v28] != v27 )
            {
              v42 = v13;
              v29 = &v27[v28];
              v30 = v12;
              do
              {
                while ( 1 )
                {
                  v31 = (unsigned __int64 *)(*v27 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (*v27 & 1) == 0 || !v31[1] )
                    break;
                  if ( v29 == ++v27 )
                    goto LABEL_34;
                }
                v32 = *v31;
                v33 = v30;
                v34 = *(_DWORD *)(v6 + 640) + 1;
                v35 = *(_QWORD **)(v6 + 608);
                v44 = *v31;
                *(_DWORD *)(v6 + 640) = v34;
                if ( !v35 )
                  goto LABEL_32;
                do
                {
                  while ( 1 )
                  {
                    v36 = v35[2];
                    v37 = v35[3];
                    if ( v32 <= v35[4] )
                      break;
                    v35 = (_QWORD *)v35[3];
                    if ( !v37 )
                      goto LABEL_30;
                  }
                  v33 = (__int64)v35;
                  v35 = (_QWORD *)v35[2];
                }
                while ( v36 );
LABEL_30:
                if ( v30 == v33 || v32 < *(_QWORD *)(v33 + 32) )
                {
LABEL_32:
                  v45[0] = &v44;
                  v33 = sub_A28390((_QWORD *)(v6 + 592), (_QWORD *)v33, v45);
                }
                ++v27;
                *(_DWORD *)(v33 + 40) = v34;
              }
              while ( v29 != v27 );
LABEL_34:
              v12 = v30;
              v13 = v42;
            }
          }
          ++v13;
        }
        while ( v43 != v13 );
        v8 = v12;
      }
      result = sub_220EF30(i);
    }
  }
  return result;
}
