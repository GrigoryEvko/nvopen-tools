// Function: sub_1832530
// Address: 0x1832530
//
void __fastcall sub_1832530(_DWORD *a1, char *a2, __int64 a3, __int64 a4)
{
  _DWORD *v4; // r8
  __int64 v6; // r9
  char *v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  char *v13; // r10
  __int64 v14; // r13
  __int64 v15; // rcx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r11
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // r15
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // r12
  _DWORD *v23; // r15
  __int64 v24; // r11
  size_t v25; // rax
  char *v26; // r15
  void *v27; // rdi
  __int64 v28; // [rsp+0h] [rbp-60h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  size_t v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  _DWORD *v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  char *v36; // [rsp+18h] [rbp-48h]
  char *v37; // [rsp+20h] [rbp-40h]
  char *v38; // [rsp+20h] [rbp-40h]
  _DWORD *v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+28h] [rbp-38h]
  _DWORD *v41; // [rsp+28h] [rbp-38h]
  _DWORD *v42; // [rsp+28h] [rbp-38h]
  unsigned __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]

  v4 = a1;
  v6 = a3;
  v8 = a2;
  v9 = a3;
  v10 = (unsigned int)a1[2];
  v11 = *(_QWORD *)a1;
  v12 = 8 * v10;
  v13 = &a2[-v11];
  v14 = v11 + 8 * v10;
  if ( v8 == (char *)v14 )
  {
    if ( v6 == a4 )
    {
      LODWORD(v22) = 0;
    }
    else
    {
      v21 = v6;
      v22 = 0;
      do
      {
        v21 = *(_QWORD *)(v21 + 8);
        ++v22;
      }
      while ( v21 != a4 );
      v23 = v4;
      if ( (unsigned int)v4[3] - v10 < v22 )
      {
        v42 = v4;
        sub_16CD150((__int64)v4, v4 + 4, v10 + v22, 8, (int)v4, v6);
        v23 = v42;
        v14 = *(_QWORD *)v42 + 8LL * (unsigned int)v42[2];
      }
      do
      {
        v14 += 8;
        *(_QWORD *)(v14 - 8) = sub_1648700(v9);
        v9 = *(_QWORD *)(v9 + 8);
      }
      while ( v9 != a4 );
      LODWORD(v10) = v23[2];
      v4 = v23;
    }
    v4[2] = v22 + v10;
  }
  else
  {
    if ( v6 == a4 )
    {
      v17 = v10;
      v16 = 0;
    }
    else
    {
      v15 = v6;
      v16 = 0;
      do
      {
        v15 = *(_QWORD *)(v15 + 8);
        ++v16;
      }
      while ( v15 != a4 );
      v17 = v10 + v16;
    }
    if ( (unsigned int)v4[3] < v17 )
    {
      v31 = v6;
      v34 = v16;
      v37 = v13;
      v41 = v4;
      sub_16CD150((__int64)v4, v4 + 4, v17, 8, (int)v4, v6);
      v4 = v41;
      v13 = v37;
      v6 = v31;
      v16 = v34;
      v10 = (unsigned int)v41[2];
      v11 = *(_QWORD *)v41;
      v12 = 8 * v10;
      v8 = &v37[*(_QWORD *)v41];
      v14 = *(_QWORD *)v41 + 8 * v10;
    }
    v18 = (v12 - (__int64)v13) >> 3;
    v19 = v18;
    if ( v16 <= v18 )
    {
      v24 = 8 * (v10 - v16);
      v25 = v12 - v24;
      v26 = (char *)(v11 + v24);
      v27 = (void *)v14;
      v43 = (v12 - v24) >> 3;
      if ( v43 > (unsigned int)v4[3] - v10 )
      {
        v28 = v6;
        v30 = v12 - v24;
        v33 = v24;
        v36 = v13;
        v39 = v4;
        sub_16CD150((__int64)v4, v4 + 4, ((v12 - v24) >> 3) + v10, 8, (int)v4, v6);
        v4 = v39;
        v6 = v28;
        v25 = v30;
        v24 = v33;
        v10 = (unsigned int)v39[2];
        v13 = v36;
        v27 = (void *)(*(_QWORD *)v39 + 8 * v10);
      }
      if ( v26 != (char *)v14 )
      {
        v29 = v6;
        v32 = v4;
        v35 = v24;
        v38 = v13;
        memmove(v27, v26, v25);
        v4 = v32;
        v6 = v29;
        v24 = v35;
        v13 = v38;
        LODWORD(v10) = v32[2];
      }
      v4[2] = v43 + v10;
      if ( v26 != v8 )
      {
        v44 = v6;
        memmove((void *)(v14 - (v24 - (_QWORD)v13)), v8, v24 - (_QWORD)v13);
        v6 = v44;
      }
      if ( v6 != a4 )
      {
        do
        {
          v8 += 8;
          *((_QWORD *)v8 - 1) = sub_1648700(v9);
          v9 = *(_QWORD *)(v9 + 8);
        }
        while ( v9 != a4 );
      }
    }
    else
    {
      v20 = v10 + v16;
      v4[2] = v20;
      if ( (char *)v14 != v8 )
      {
        v40 = (v12 - (__int64)v13) >> 3;
        memcpy((void *)(v11 + 8LL * v20 - (v12 - (_QWORD)v13)), v8, v12 - (_QWORD)v13);
        v18 = v40;
      }
      if ( v18 )
      {
        do
        {
          v8 += 8;
          *((_QWORD *)v8 - 1) = sub_1648700(v9);
          v9 = *(_QWORD *)(v9 + 8);
          --v19;
        }
        while ( v19 );
      }
      for ( ; v9 != a4; v9 = *(_QWORD *)(v9 + 8) )
      {
        v14 += 8;
        *(_QWORD *)(v14 - 8) = sub_1648700(v9);
      }
    }
  }
}
