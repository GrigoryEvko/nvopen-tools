// Function: sub_1DBAB50
// Address: 0x1dbab50
//
void __fastcall sub_1DBAB50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 j; // rax
  _WORD *v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // ecx
  __int16 v14; // ax
  _WORD *v15; // rcx
  _WORD *v16; // rdx
  unsigned __int16 v17; // r14
  _WORD *v18; // r13
  __int16 v19; // ax
  __int64 *v20; // r12
  __int64 *v21; // rax
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int *v27; // rbx
  unsigned int *v28; // r12
  __int64 v29; // rcx
  __int64 v30; // rdx
  _QWORD *v31; // rax
  _QWORD *i; // rdx
  __int64 v33; // [rsp+10h] [rbp-90h]
  _WORD *v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+20h] [rbp-80h]
  _WORD *v36; // [rsp+28h] [rbp-78h]
  char v37; // [rsp+38h] [rbp-68h]
  unsigned int *v38; // [rsp+40h] [rbp-60h] BYREF
  __int64 v39; // [rsp+48h] [rbp-58h]
  _BYTE v40[80]; // [rsp+50h] [rbp-50h] BYREF

  v6 = *(unsigned int *)(*(_QWORD *)(a1 + 248) + 44LL);
  v7 = *(unsigned int *)(a1 + 680);
  if ( v6 >= v7 )
  {
    if ( v6 <= v7 )
      goto LABEL_3;
    if ( v6 > *(unsigned int *)(a1 + 684) )
    {
      sub_16CD150(a1 + 672, (const void *)(a1 + 688), v6, 8, a5, a6);
      v7 = *(unsigned int *)(a1 + 680);
    }
    v30 = *(_QWORD *)(a1 + 672);
    v31 = (_QWORD *)(v30 + 8 * v7);
    for ( i = (_QWORD *)(v30 + 8 * v6); i != v31; ++v31 )
    {
      if ( v31 )
        *v31 = 0;
    }
  }
  *(_DWORD *)(a1 + 680) = v6;
LABEL_3:
  v38 = (unsigned int *)v40;
  v39 = 0x800000000LL;
  v8 = *(_QWORD *)(a1 + 232);
  v33 = v8 + 320;
  v35 = *(_QWORD *)(v8 + 328);
  if ( v35 != v8 + 320 )
  {
    for ( j = *(_QWORD *)(v8 + 328); ; j = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 328LL) )
    {
      if ( v35 == j || *(_BYTE *)(v35 + 180) )
      {
        v10 = *(_WORD **)(v35 + 160);
        v34 = v10;
        if ( v10 != *(_WORD **)(v35 + 152) )
        {
          v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL) + 16LL * *(unsigned int *)(v35 + 48));
          v36 = (_WORD *)sub_1DD77D0();
          if ( v10 != v36 )
          {
            do
            {
              v12 = *(_QWORD *)(a1 + 248);
              if ( !v12 )
                BUG();
              v13 = *(_DWORD *)(*(_QWORD *)(v12 + 8) + 24LL * (unsigned __int16)*v36 + 16);
              v14 = v13 & 0xF;
              v15 = (_WORD *)(*(_QWORD *)(v12 + 56) + 2LL * (v13 >> 4));
              v16 = v15 + 1;
              v17 = *v15 + v14 * *v36;
LABEL_14:
              v18 = v16;
              while ( v18 )
              {
                v20 = *(__int64 **)(*(_QWORD *)(a1 + 672) + 8LL * v17);
                if ( !v20 )
                {
                  v37 = qword_4FC4440[20];
                  v21 = (__int64 *)sub_22077B0(104);
                  v24 = v17;
                  v20 = v21;
                  if ( v21 )
                  {
                    *v21 = (__int64)(v21 + 2);
                    v21[1] = 0x200000000LL;
                    v21[8] = (__int64)(v21 + 10);
                    v21[9] = 0x200000000LL;
                    if ( v37 )
                    {
                      v26 = sub_22077B0(48);
                      v24 = v17;
                      if ( v26 )
                      {
                        *(_DWORD *)(v26 + 8) = 0;
                        *(_QWORD *)(v26 + 16) = 0;
                        *(_QWORD *)(v26 + 24) = v26 + 8;
                        *(_QWORD *)(v26 + 32) = v26 + 8;
                        *(_QWORD *)(v26 + 40) = 0;
                      }
                      v20[12] = v26;
                    }
                    else
                    {
                      v21[12] = 0;
                    }
                  }
                  *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v24) = v20;
                  v25 = (unsigned int)v39;
                  if ( (unsigned int)v39 >= HIDWORD(v39) )
                  {
                    sub_16CD150((__int64)&v38, v40, 0, 4, v22, v23);
                    v25 = (unsigned int)v39;
                  }
                  v38[v25] = v17;
                  LODWORD(v39) = v39 + 1;
                }
                ++v18;
                sub_1DB79D0(v20, v11, (__int64 *)(a1 + 296));
                v19 = *(v18 - 1);
                v16 = 0;
                v17 += v19;
                if ( !v19 )
                  goto LABEL_14;
              }
              v36 += 4;
            }
            while ( v34 != v36 );
          }
        }
      }
      v35 = *(_QWORD *)(v35 + 8);
      if ( v33 == v35 )
        break;
    }
    v27 = v38;
    v28 = &v38[(unsigned int)v39];
    if ( v38 != v28 )
    {
      do
      {
        v29 = *v27++;
        sub_1DBA8F0((_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v29), v29);
      }
      while ( v28 != v27 );
      v28 = v38;
    }
    if ( v28 != (unsigned int *)v40 )
      _libc_free((unsigned __int64)v28);
  }
}
