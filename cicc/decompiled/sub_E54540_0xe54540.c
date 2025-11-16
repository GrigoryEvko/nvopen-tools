// Function: sub_E54540
// Address: 0xe54540
//
__int64 __fastcall sub_E54540(
        __int64 a1,
        unsigned int a2,
        char *a3,
        __int64 a4,
        unsigned __int8 *a5,
        __int64 a6,
        unsigned int a7)
{
  int v11; // eax
  __int64 v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdi
  _BYTE *v18; // rax
  unsigned __int8 *v19; // rax
  __int64 v20; // r15
  __int64 v21; // rbx
  _BYTE *v22; // r8
  unsigned __int64 v23; // r13
  unsigned __int8 *v24; // rdx
  unsigned __int8 *v25; // r9
  __int64 v26; // rcx
  unsigned __int8 v27; // al
  _BYTE *v28; // rsi
  __int64 v29; // rdi
  _BYTE *v30; // rax
  __int128 v31; // [rsp-10h] [rbp-A0h]
  unsigned __int8 v32; // [rsp+7h] [rbp-89h]
  char *v34[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v35[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v36; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h]
  _BYTE v39[72]; // [rsp+48h] [rbp-48h] BYREF

  v11 = sub_E66210(*(_QWORD *)(a1 + 8));
  *((_QWORD *)&v31 + 1) = a6;
  *(_QWORD *)&v31 = a5;
  v32 = sub_E5F990(v11, a1, a2, (_DWORD)a3, a4, (unsigned __int8)a7, v31);
  if ( v32 )
  {
    v13 = sub_904010(*(_QWORD *)(a1 + 304), "\t.cv_file\t");
    v14 = sub_CB59D0(v13, a2);
    v15 = *(_BYTE **)(v14 + 32);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
    {
      sub_CB5D20(v14, 32);
    }
    else
    {
      *(_QWORD *)(v14 + 32) = v15 + 1;
      *v15 = 32;
    }
    sub_E51560(a1, a3, a4, *(_QWORD *)(a1 + 304));
    if ( a7 )
    {
      v17 = *(_QWORD *)(a1 + 304);
      v18 = *(_BYTE **)(v17 + 32);
      if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
      {
        sub_CB5D20(v17, 32);
      }
      else
      {
        *(_QWORD *)(v17 + 32) = v18 + 1;
        *v18 = 32;
      }
      v19 = a5;
      v37 = 0;
      v20 = *(_QWORD *)(a1 + 304);
      v36 = v39;
      v21 = a6;
      v22 = v39;
      v38 = 16;
      v23 = 2 * a6;
      if ( v23 )
      {
        if ( v23 > 0x10 )
        {
          sub_C8D290((__int64)&v36, v39, v23, 1u, (__int64)v39, v16);
          v22 = v36;
          v19 = a5;
        }
        v37 = v23;
      }
      if ( v21 )
      {
        v24 = v19;
        v25 = &v19[v21];
        v26 = 0;
        do
        {
          v27 = *v24++;
          v22[v26] = a0123456789abcd_10[v27 >> 4];
          v36[v26 + 1] = a0123456789abcd_10[v27 & 0xF];
          v26 += 2;
          v22 = v36;
        }
        while ( v25 != v24 );
      }
      v28 = v22;
      v34[0] = (char *)v35;
      sub_E4CC80((__int64 *)v34, v22, (__int64)&v22[v37]);
      if ( v36 != v39 )
        _libc_free(v36, v28);
      sub_E51560(a1, v34[0], (__int64)v34[1], v20);
      if ( (_QWORD *)v34[0] != v35 )
        j_j___libc_free_0(v34[0], v35[0] + 1LL);
      v29 = *(_QWORD *)(a1 + 304);
      v30 = *(_BYTE **)(v29 + 32);
      if ( (unsigned __int64)v30 >= *(_QWORD *)(v29 + 24) )
      {
        v29 = sub_CB5D20(v29, 32);
      }
      else
      {
        *(_QWORD *)(v29 + 32) = v30 + 1;
        *v30 = 32;
      }
      sub_CB59D0(v29, a7);
    }
    sub_E4D880(a1);
  }
  return v32;
}
