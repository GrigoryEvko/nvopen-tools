// Function: sub_24B12F0
// Address: 0x24b12f0
//
void __fastcall sub_24B12F0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rdi
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  int v6; // r12d
  unsigned int i; // r14d
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r10
  __int64 v15; // rax
  int v16; // r15d
  unsigned int v17; // ebx
  char *v18; // rsi
  char *v19; // rsi
  char *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  bool v24; // zf
  int v25; // edx
  int v26; // r8d
  const char *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r12
  unsigned __int8 *v32; // rax
  size_t v33; // rdx
  void *v34; // rdi
  size_t v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // [rsp+0h] [rbp-90h]
  __int64 v41; // [rsp+8h] [rbp-88h]
  _QWORD *v42; // [rsp+18h] [rbp-78h]
  unsigned int v43; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v44; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v45[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v46; // [rsp+40h] [rbp-50h] BYREF
  char *v47; // [rsp+48h] [rbp-48h]
  char *v48; // [rsp+50h] [rbp-40h]

  v1 = a1;
  v2 = *a1;
  v46 = 0;
  v47 = 0;
  v3 = *(_QWORD *)(v2 + 80);
  v48 = 0;
  v43 = -1;
  v41 = v3;
  v40 = v2 + 72;
  if ( v3 == v2 + 72 )
  {
    v20 = 0;
    v19 = 0;
  }
  else
  {
    do
    {
      if ( !v41 )
        BUG();
      v4 = *(_QWORD *)(v41 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 != v41 + 24 )
      {
        if ( !v4 )
          BUG();
        v5 = v4 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 <= 0xA )
        {
          v6 = sub_B46E30(v5);
          if ( v6 )
          {
            for ( i = 0; i != v6; ++i )
            {
              v8 = sub_B46EC0(v5, i);
              v9 = v1[31];
              v10 = v8;
              v11 = *((unsigned int *)v1 + 66);
              if ( (_DWORD)v11 )
              {
                v12 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
                v13 = (__int64 *)(v9 + 16LL * v12);
                v14 = *v13;
                if ( v10 == *v13 )
                {
LABEL_10:
                  if ( v13 != (__int64 *)(v9 + 16 * v11) )
                  {
                    v15 = v13[1];
                    if ( v15 )
                    {
                      v42 = v1;
                      v16 = 0;
                      v17 = *(_DWORD *)(v15 + 8);
                      do
                      {
                        v18 = v47;
                        LOBYTE(v45[0]) = v17 >> v16;
                        if ( v47 == v48 )
                        {
                          sub_C8FB10((__int64)&v46, v47, (char *)v45);
                        }
                        else
                        {
                          if ( v47 )
                          {
                            *v47 = v17 >> v16;
                            v18 = v47;
                          }
                          v47 = v18 + 1;
                        }
                        v16 += 8;
                      }
                      while ( v16 != 32 );
                      v1 = v42;
                    }
                  }
                }
                else
                {
                  v25 = 1;
                  while ( v14 != -4096 )
                  {
                    v26 = v25 + 1;
                    v12 = (v11 - 1) & (v25 + v12);
                    v13 = (__int64 *)(v9 + 16LL * v12);
                    v14 = *v13;
                    if ( v10 == *v13 )
                      goto LABEL_10;
                    v25 = v26;
                  }
                }
              }
            }
          }
        }
      }
      v41 = *(_QWORD *)(v41 + 8);
    }
    while ( v40 != v41 );
    v19 = (char *)v46;
    v20 = &v47[-v46];
  }
  sub_1098F90(&v43, v19, (__int64)v20);
  v21 = *((unsigned int *)v1 + 18);
  v44 = -1;
  v45[0] = v21;
  sub_1098F90(&v44, (char *)v45, 8);
  v45[0] = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v1[5] + 8LL) - *(_QWORD *)v1[5]) >> 3);
  sub_1098F90(&v44, (char *)v45, 8);
  v45[0] = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v1[5] + 32LL) - *(_QWORD *)(v1[5] + 24LL)) >> 3);
  sub_1098F90(&v44, (char *)v45, 8);
  if ( *((_BYTE *)v1 + 392) )
    v22 = sub_3158260(v1 + 39);
  else
    v22 = (__int64)(v1[28] - v1[27]) >> 3;
  v45[0] = v22;
  sub_1098F90(&v44, (char *)v45, 8);
  v23 = (v43 + ((unsigned __int64)v44 << 28)) & 0xFFFFFFFFFFFFFFFLL;
  v24 = *((_BYTE *)v1 + 8) == 0;
  v1[25] = v23;
  if ( !v24 )
    v1[25] = v23 | 0x1000000000000000LL;
  if ( sub_2241AC0((__int64)&qword_4FEADC8, "-") )
  {
    v27 = sub_BD5D20(*v1);
    v28 = qword_4FEADC8;
    v45[1] = v29;
    v45[0] = (__int64)v27;
    if ( sub_C931B0(v45, (_WORD *)qword_4FEADC8, qword_4FEADD0, 0) != -1 )
    {
      v30 = sub_C5F790((__int64)v45, v28);
      v31 = sub_904010(v30, "Funcname=");
      v32 = (unsigned __int8 *)sub_BD5D20(*v1);
      v34 = *(void **)(v31 + 32);
      v35 = v33;
      if ( v33 > *(_QWORD *)(v31 + 24) - (_QWORD)v34 )
      {
        v31 = sub_CB6200(v31, v32, v33);
      }
      else if ( v33 )
      {
        memcpy(v34, v32, v33);
        *(_QWORD *)(v31 + 32) += v35;
      }
      v36 = sub_904010(v31, ", Hash=");
      v37 = sub_CB59D0(v36, v1[25]);
      v38 = sub_904010(v37, " in building ");
      v39 = sub_CB6200(
              v38,
              *(unsigned __int8 **)(*(_QWORD *)(*v1 + 40LL) + 200LL),
              *(_QWORD *)(*(_QWORD *)(*v1 + 40LL) + 208LL));
      sub_904010(v39, "\n");
    }
  }
  if ( v46 )
    j_j___libc_free_0(v46);
}
