// Function: sub_32577A0
// Address: 0x32577a0
//
__int64 __fastcall sub_32577A0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, _QWORD *a5)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v9; // rbx
  __int64 v11; // r13
  __int64 v12; // rsi
  int v13; // edx
  __int64 v14; // rsi
  __int64 *v15; // r12
  __int64 v16; // rcx
  int v17; // edx
  __int64 *v18; // r10
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r9
  int v24; // r10d
  unsigned __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // r10
  unsigned int v29; // r11d
  unsigned int v30; // eax
  int v31; // r10d
  int v32; // r11d
  _QWORD *v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  int v35; // [rsp+10h] [rbp-50h]
  int v36; // [rsp+14h] [rbp-4Ch]
  unsigned int v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 v38; // [rsp+18h] [rbp-48h]
  unsigned int v39; // [rsp+20h] [rbp-40h]
  const void *v40; // [rsp+28h] [rbp-38h]

  result = *(_QWORD *)(a1 + 8);
  v6 = *((unsigned int *)a4 + 2);
  v40 = (const void *)(a2 + 16);
  v7 = *(_QWORD *)(result + 232);
  if ( (_DWORD)v6 )
  {
    v9 = 0;
    v11 = 4 * v6;
    do
    {
      result = *a4;
      v13 = *(_DWORD *)(v7 + 512);
      v14 = *(_QWORD *)(v7 + 496);
      v15 = *(__int64 **)(*a4 + 2 * v9);
      v16 = *v15;
      if ( !v13 )
        goto LABEL_5;
      v17 = v13 - 1;
      result = v17 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v18 = (__int64 *)(v14 + 16 * result);
      v19 = *v18;
      if ( v16 != *v18 )
      {
        v39 = 1;
        v28 = *v18;
        v29 = result;
        v37 = result;
        while ( v28 != -4096 )
        {
          result = v39;
          v29 = v17 & (v39 + v29);
          ++v39;
          v28 = *(_QWORD *)(v14 + 16LL * v29);
          if ( v16 == v28 )
          {
            v30 = v37;
            v31 = 1;
            while ( v19 != -4096 )
            {
              v32 = v31 + 1;
              v30 = v17 & (v31 + v30);
              v18 = (__int64 *)(v14 + 16LL * v30);
              v19 = *v18;
              if ( v16 == *v18 )
                goto LABEL_8;
              v31 = v32;
            }
            v22 = 0;
            v21 = 1;
            v23 = 1;
            goto LABEL_9;
          }
        }
        goto LABEL_5;
      }
LABEL_8:
      v20 = *((unsigned int *)v18 + 2);
      v21 = (unsigned int)(v20 + 1);
      v22 = 32 * v20;
      v23 = v21;
LABEL_9:
      v24 = *(_DWORD *)(*a5 + v9);
      v25 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v25 < (unsigned int)v23 && v21 != v25 )
      {
        if ( v21 >= v25 )
        {
          if ( v21 > *(unsigned int *)(a2 + 12) )
          {
            v33 = a5;
            v35 = v23;
            v34 = v22;
            v36 = *(_DWORD *)(*a5 + v9);
            v38 = v21;
            sub_C8D5F0(a2, v40, v21, 0x20u, (__int64)a5, v23);
            a5 = v33;
            LODWORD(v23) = v35;
            v22 = v34;
            v25 = *(unsigned int *)(a2 + 8);
            v24 = v36;
            v21 = v38;
          }
          v12 = *(_QWORD *)a2;
          v26 = *(_QWORD *)a2 + 32 * v25;
          v27 = *(_QWORD *)a2 + 32 * v21;
          if ( v26 != v27 )
          {
            do
            {
              if ( v26 )
              {
                *(_QWORD *)v26 = 0;
                *(_QWORD *)(v26 + 8) = 0;
                *(_QWORD *)(v26 + 16) = 0;
                *(_DWORD *)(v26 + 24) = 0;
              }
              v26 += 32;
            }
            while ( v27 != v26 );
            v12 = *(_QWORD *)a2;
          }
          *(_DWORD *)(a2 + 8) = v23;
          goto LABEL_4;
        }
        *(_DWORD *)(a2 + 8) = v23;
      }
      v12 = *(_QWORD *)a2;
LABEL_4:
      result = v12 + v22;
      *(_QWORD *)result = 0;
      *(_QWORD *)(result + 8) = 0;
      *(_QWORD *)(result + 16) = v15;
      *(_DWORD *)(result + 24) = v24;
LABEL_5:
      v9 += 4;
    }
    while ( v9 != v11 );
  }
  return result;
}
