// Function: sub_E37DE0
// Address: 0xe37de0
//
__int64 __fastcall sub_E37DE0(_DWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  _DWORD *v6; // r10
  int v7; // r14d
  unsigned int v9; // r12d
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // r11
  unsigned __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  __int64 *v19; // r15
  __int64 v20; // rcx
  __int64 result; // rax
  unsigned int v22; // r14d
  __int64 *v23; // r12
  unsigned int v24; // esi
  __int64 *v25; // r13
  _DWORD *v26; // r15
  __int64 v27; // r14
  __int64 *v28; // r12
  signed __int64 v29; // r11
  __int64 *v30; // rdi
  unsigned __int64 v31; // rdx
  unsigned int v32; // r12d
  int v33; // r14d
  unsigned int v34; // esi
  unsigned __int64 v35; // [rsp+0h] [rbp-60h]
  signed __int64 v36; // [rsp+0h] [rbp-60h]
  unsigned int v37; // [rsp+8h] [rbp-58h]
  _DWORD *v38; // [rsp+8h] [rbp-58h]
  int v39; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  int v41; // [rsp+10h] [rbp-50h]
  int v42; // [rsp+10h] [rbp-50h]
  unsigned __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  _DWORD *v47; // [rsp+18h] [rbp-48h]
  unsigned __int64 v48; // [rsp+20h] [rbp-40h]
  _DWORD *v49; // [rsp+20h] [rbp-40h]
  __int64 v50; // [rsp+20h] [rbp-40h]
  int v51; // [rsp+20h] [rbp-40h]
  _DWORD *v52; // [rsp+20h] [rbp-40h]
  unsigned __int64 v54; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v7 = a6;
  v9 = a4;
  v11 = (unsigned int)(a6 - a4);
  v12 = (unsigned int)a1[2];
  v13 = *(_QWORD *)a1;
  v14 = (int)v11;
  v15 = (unsigned int)a1[3];
  v16 = 8 * v12;
  v17 = (__int64)a2 - v13;
  v18 = v12 + (int)v11;
  v19 = (__int64 *)(v13 + 8 * v12);
  if ( a2 == v19 )
  {
    if ( v18 > v15 )
    {
      v43 = (int)v11;
      v52 = v6;
      sub_C8D5F0((__int64)v6, v6 + 4, v18, 8u, v11, v17);
      v6 = v52;
      v14 = v43;
      v12 = (unsigned int)v52[2];
      v19 = (__int64 *)(*(_QWORD *)v52 + 8 * v12);
    }
    if ( (_DWORD)a4 != a6 )
    {
      v54 = v14;
      v25 = v19;
      v26 = v6;
      do
      {
        if ( v25 )
          *v25 = sub_B46EC0(a3, v9);
        ++v9;
        ++v25;
      }
      while ( v9 != v7 );
      v14 = v54;
      v12 = (unsigned int)v26[2];
      v6 = v26;
    }
    result = v14 + v12;
    v6[2] = result;
  }
  else
  {
    if ( v18 > v15 )
    {
      v35 = (int)v11;
      v37 = v11;
      v49 = v6;
      sub_C8D5F0((__int64)v6, v6 + 4, v18, 8u, v11, v17);
      v6 = v49;
      v17 = (__int64)a2 - v13;
      v14 = v35;
      v11 = v37;
      v12 = (unsigned int)v49[2];
      v13 = *(_QWORD *)v49;
      v16 = 8 * v12;
      a2 = (__int64 *)(*(_QWORD *)v49 + v17);
      v19 = (__int64 *)(*(_QWORD *)v49 + 8 * v12);
    }
    v20 = v16 - v17;
    v48 = (v16 - v17) >> 3;
    if ( v48 >= v14 )
    {
      v27 = 8 * (v12 - v14);
      v28 = (__int64 *)(v13 + v27);
      v29 = v16 - v27;
      v30 = v19;
      v50 = v29 >> 3;
      v31 = v12 + (v29 >> 3);
      if ( v31 > (unsigned int)v6[3] )
      {
        v36 = v29;
        v39 = v11;
        v44 = v17;
        v47 = v6;
        sub_C8D5F0((__int64)v6, v6 + 4, v31, 8u, v11, v17);
        v6 = v47;
        v29 = v36;
        LODWORD(v11) = v39;
        v17 = v44;
        v12 = (unsigned int)v47[2];
        v30 = (__int64 *)(*(_QWORD *)v47 + 8 * v12);
      }
      if ( v28 != v19 )
      {
        v38 = v6;
        v42 = v11;
        v46 = v17;
        memmove(v30, v28, v29);
        v6 = v38;
        LODWORD(v11) = v42;
        v17 = v46;
        LODWORD(v12) = v38[2];
      }
      v6[2] = v50 + v12;
      if ( v28 != a2 )
      {
        v51 = v11;
        memmove((char *)v19 - (v27 - v17), a2, v27 - v17);
        LODWORD(v11) = v51;
      }
      result = a4;
      v32 = a4;
      v33 = v11 + a4;
      if ( (int)v11 > 0 )
      {
        do
        {
          v34 = v32++;
          ++a2;
          result = sub_B46EC0(a3, v34);
          *(a2 - 1) = result;
        }
        while ( v33 != v32 );
      }
    }
    else
    {
      result = v14 + v12;
      v6[2] = result;
      if ( a2 != v19 )
      {
        v40 = v16 - v17;
        result = (__int64)memcpy((void *)(v13 + 8LL * (unsigned int)result - v20), a2, v16 - v17);
        v20 = v40;
      }
      if ( v48 )
      {
        v41 = v7;
        v22 = a4;
        v23 = (__int64 *)((char *)a2 + v20);
        do
        {
          v24 = v22;
          ++a2;
          ++v22;
          *(a2 - 1) = sub_B46EC0(a3, v24);
        }
        while ( a2 != v23 );
        result = a4;
        v7 = v41;
        v9 = a4 + v48;
      }
      if ( v9 != a6 )
      {
        do
        {
          if ( v19 )
          {
            result = sub_B46EC0(a3, v9);
            *v19 = result;
          }
          ++v9;
          ++v19;
        }
        while ( v7 != v9 );
      }
    }
  }
  return result;
}
