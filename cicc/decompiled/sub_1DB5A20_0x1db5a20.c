// Function: sub_1DB5A20
// Address: 0x1db5a20
//
__int64 __fastcall sub_1DB5A20(_DWORD *a1, __int64 a2)
{
  unsigned int **v4; // r12
  unsigned int *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 *v8; // rdx
  unsigned int *v9; // r13
  unsigned __int64 v10; // r9
  unsigned int *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 *v18; // rsi
  unsigned int *v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // eax
  __int64 *v23; // r10
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 *v27; // rsi
  unsigned int **v29; // [rsp+8h] [rbp-58h]
  unsigned int *v30; // [rsp+10h] [rbp-50h]
  unsigned int **v31; // [rsp+18h] [rbp-48h]
  _DWORD *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]
  __int64 *v34; // [rsp+28h] [rbp-38h]

  a1[4] = 0;
  a1[14] = 0;
  v32 = a1 + 2;
  sub_3945AE0(a1 + 2, *(unsigned int *)(a2 + 72));
  v4 = *(unsigned int ***)(a2 + 64);
  v30 = 0;
  v31 = &v4[*(unsigned int *)(a2 + 72)];
  if ( v4 == v31 )
    goto LABEL_14;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v9 = *v4;
      v10 = *((_QWORD *)*v4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v10 )
        break;
      v6 = (*((__int64 *)*v4 + 1) >> 1) & 3;
      if ( !(_DWORD)v6 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)a1 + 272LL);
        v14 = *(_QWORD *)(v10 + 16);
        if ( v14 )
        {
          v15 = *(_QWORD *)(v14 + 24);
          goto LABEL_19;
        }
        v23 = *(__int64 **)(v13 + 536);
        v24 = *(unsigned int *)(v13 + 544);
        v34 = &v23[2 * v24];
        v25 = (16 * v24) >> 4;
        if ( 16 * v24 )
        {
          do
          {
            while ( 1 )
            {
              v26 = v25 >> 1;
              v27 = &v23[2 * (v25 >> 1)];
              if ( (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) >= *(_DWORD *)(v10 + 24) )
                break;
              v23 = v27 + 2;
              v25 = v25 - v26 - 1;
              if ( v25 <= 0 )
                goto LABEL_37;
            }
            v25 >>= 1;
          }
          while ( v26 > 0 );
        }
LABEL_37:
        if ( v34 == v23 )
        {
          if ( !(_DWORD)v24 )
            goto LABEL_40;
        }
        else if ( (*(_DWORD *)((*v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v23 >> 1) & 3) <= *(_DWORD *)(v10 + 24) )
        {
LABEL_40:
          v15 = v23[1];
LABEL_19:
          v33 = *(_QWORD *)(v15 + 72);
          if ( v33 != *(_QWORD *)(v15 + 64) )
          {
            v29 = v4;
            v16 = *(_QWORD *)(v15 + 64);
            while ( 1 )
            {
              v20 = *(_QWORD *)(*(_QWORD *)(v13 + 392) + 16LL * *(unsigned int *)(*(_QWORD *)v16 + 48LL) + 8);
              v21 = v20 & 0xFFFFFFFFFFFFFFF8LL;
              v22 = (v20 >> 1) & 3;
              if ( v22 )
                v17 = (2LL * (v22 - 1)) | v21;
              else
                v17 = *(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL | 6;
              v18 = (__int64 *)sub_1DB3C70((__int64 *)a2, v17);
              if ( v18 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
                && (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) <= (*(_DWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v17 >> 1) & 3) )
              {
                v19 = (unsigned int *)v18[2];
                if ( v19 )
                  sub_3945B70(v32, *v9, *v19);
              }
              v16 += 8;
              if ( v33 == v16 )
                break;
              v13 = *(_QWORD *)(*(_QWORD *)a1 + 272LL);
            }
            v4 = v29;
          }
          goto LABEL_6;
        }
        v23 -= 2;
        goto LABEL_40;
      }
      v7 = v10 | (2LL * ((int)v6 - 1));
      v8 = (__int64 *)sub_1DB3C70((__int64 *)a2, v7);
      if ( v8 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
        && (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3) <= (*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v7 >> 1)
                                                                                             & 3) )
      {
        v12 = (unsigned int *)v8[2];
        if ( v12 )
        {
          v5 = v9;
          sub_3945B70(v32, *v9, *v12);
          goto LABEL_7;
        }
      }
LABEL_6:
      v5 = v9;
LABEL_7:
      if ( v31 == ++v4 )
        goto LABEL_11;
    }
    if ( !v30 )
    {
      v30 = *v4;
      goto LABEL_7;
    }
    ++v4;
    sub_3945B70(v32, *v30, *v9);
    v30 = v9;
  }
  while ( v31 != v4 );
LABEL_11:
  if ( v5 && v30 )
    sub_3945B70(v32, *v5, *v30);
LABEL_14:
  sub_3945BD0(v32);
  return (unsigned int)a1[14];
}
