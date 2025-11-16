// Function: sub_17AE390
// Address: 0x17ae390
//
__int64 __fastcall sub_17AE390(_BYTE *a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  __int64 v6; // r12
  char v7; // al
  __int64 v8; // r8
  unsigned int v9; // r15d
  __int64 v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // r14
  __int64 v14; // rdi
  char v15; // al
  __int64 v16; // rax
  _BYTE *v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // r14
  _QWORD *v22; // rax
  __int64 v23; // rax
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // r15
  __int64 v27; // rax
  int v28; // r12d
  const void *v29; // r15
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r9d
  unsigned int v34; // r8d
  __int64 v35; // r14
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 **v38; // rax
  __int64 v39; // rax
  int v40; // r8d
  int v41; // r9d
  __int64 v42; // r13
  __int64 *v43; // rax
  __int64 *v44; // rdx
  _QWORD *v45; // rax
  __int64 **v46; // rax
  __int64 *v47; // rbx
  _QWORD *v48; // rax
  __int64 v49; // rax
  __int64 *v50; // rbx
  unsigned int v51; // [rsp+4h] [rbp-4Ch]
  __int64 v52; // [rsp+10h] [rbp-40h]
  __int64 v53; // [rsp+18h] [rbp-38h]
  const void *v54; // [rsp+18h] [rbp-38h]
  unsigned int v55; // [rsp+18h] [rbp-38h]

  v6 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v7 = a1[16];
  if ( v7 != 9 )
  {
    if ( a1 == a2 )
    {
      if ( (_DWORD)v6 )
      {
        v21 = 0;
        v54 = (const void *)(a4 + 16);
        do
        {
          v22 = (_QWORD *)sub_16498A0((__int64)a1);
          v23 = sub_1643350(v22);
          v26 = sub_159C470(v23, v21, 0);
          v27 = *(unsigned int *)(a4 + 8);
          if ( (unsigned int)v27 >= *(_DWORD *)(a4 + 12) )
          {
            sub_16CD150(a4, v54, 0, 8, v24, v25);
            v27 = *(unsigned int *)(a4 + 8);
          }
          ++v21;
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v27) = v26;
          ++*(_DWORD *)(a4 + 8);
        }
        while ( (unsigned int)v6 != v21 );
      }
    }
    else
    {
      v8 = (unsigned int)v6;
      if ( a1 != a3 )
      {
        if ( v7 == 84 )
        {
          v11 = *((_QWORD *)a1 - 3);
          if ( *(_BYTE *)(v11 + 16) == 13 )
          {
            v12 = *(_QWORD **)(v11 + 24);
            if ( *(_DWORD *)(v11 + 32) > 0x40u )
              v12 = (_QWORD *)*v12;
            v13 = *((_QWORD *)a1 - 6);
            v14 = *((_QWORD *)a1 - 9);
            v15 = *(_BYTE *)(v13 + 16);
            if ( v15 == 9 )
            {
              v9 = sub_17AE390(v14, a2, a3, a4, v8);
              if ( (_BYTE)v9 )
              {
                v45 = (_QWORD *)sub_16498A0((__int64)a1);
                v46 = (__int64 **)sub_1643350(v45);
                v47 = (__int64 *)(*(_QWORD *)a4 + 8LL * (unsigned int)v12);
                *v47 = sub_1599EF0(v46);
                return v9;
              }
            }
            else if ( v15 == 83 )
            {
              v16 = *(_QWORD *)(v13 - 24);
              if ( *(_BYTE *)(v16 + 16) == 13 )
              {
                v53 = *(_DWORD *)(v16 + 32) <= 0x40u ? *(_QWORD *)(v16 + 24) : **(_QWORD **)(v16 + 24);
                if ( (v17 = *(_BYTE **)(v13 - 48)) != 0 && a2 == v17 || a3 == v17 )
                {
                  v51 = v8;
                  v52 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
                  v9 = sub_17AE390(v14, a2, a3, a4, v8);
                  if ( (_BYTE)v9 )
                  {
                    LOBYTE(v13) = *(_QWORD *)(v13 - 48) != 0 && a2 == *(_BYTE **)(v13 - 48);
                    if ( (_BYTE)v13 )
                    {
                      v9 = v13;
                      v48 = (_QWORD *)sub_16498A0((__int64)a1);
                      v49 = sub_1643350(v48);
                      v50 = (__int64 *)(*(_QWORD *)a4 + 8LL * ((unsigned int)v12 % v51));
                      *v50 = sub_159C470(v49, (unsigned int)v53, 0);
                    }
                    else
                    {
                      v18 = (_QWORD *)sub_16498A0((__int64)a1);
                      v19 = sub_1643350(v18);
                      v20 = (__int64 *)(*(_QWORD *)a4 + 8LL * ((unsigned int)v12 % v51));
                      *v20 = sub_159C470(v19, (unsigned int)(v53 + v52), 0);
                    }
                    return v9;
                  }
                }
              }
            }
          }
        }
        return 0;
      }
      if ( (_DWORD)v6 )
      {
        v28 = 2 * v6;
        v29 = (const void *)(a4 + 16);
        do
        {
          v55 = v8;
          v30 = (_QWORD *)sub_16498A0((__int64)a1);
          v31 = sub_1643350(v30);
          v32 = sub_159C470(v31, v55, 0);
          v34 = v55;
          v35 = v32;
          v36 = *(unsigned int *)(a4 + 8);
          if ( (unsigned int)v36 >= *(_DWORD *)(a4 + 12) )
          {
            sub_16CD150(a4, v29, 0, 8, v55, v33);
            v36 = *(unsigned int *)(a4 + 8);
            v34 = v55;
          }
          LODWORD(v8) = v34 + 1;
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v36) = v35;
          ++*(_DWORD *)(a4 + 8);
        }
        while ( v28 != (_DWORD)v8 );
      }
    }
    return 1;
  }
  v37 = (_QWORD *)sub_16498A0((__int64)a1);
  v38 = (__int64 **)sub_1643350(v37);
  v39 = sub_1599EF0(v38);
  *(_DWORD *)(a4 + 8) = 0;
  v42 = v39;
  if ( *(_DWORD *)(a4 + 12) < (unsigned int)v6 )
    sub_16CD150(a4, (const void *)(a4 + 16), (unsigned int)v6, 8, v40, v41);
  v43 = *(__int64 **)a4;
  *(_DWORD *)(a4 + 8) = v6;
  v44 = &v43[(unsigned int)v6];
  if ( v43 == v44 )
    return 1;
  do
    *v43++ = v42;
  while ( v44 != v43 );
  return 1;
}
