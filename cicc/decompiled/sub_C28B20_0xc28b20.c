// Function: sub_C28B20
// Address: 0xc28b20
//
__int64 __fastcall sub_C28B20(_QWORD *a1, unsigned __int8 a2)
{
  __int64 result; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rdi
  _QWORD *v11; // r9
  __int64 v12; // r12
  _QWORD *v13; // rax
  _QWORD *v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // r9
  __int64 v17; // r13
  __int64 v18; // r12
  _QWORD *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  char v22; // al
  unsigned __int64 v23; // rdx
  _QWORD *v24; // r9
  unsigned __int64 v25; // rcx
  _QWORD *v26; // r11
  _QWORD **v27; // rax
  _QWORD *v28; // rdx
  size_t v29; // r12
  void *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // r8
  _QWORD *v33; // rdi
  unsigned __int64 v34; // r10
  _QWORD *v35; // rsi
  unsigned __int64 v36; // rdx
  _QWORD **v37; // rax
  _QWORD *v38; // rdi
  __int64 v39; // rdx
  unsigned __int64 v40; // [rsp+8h] [rbp-158h]
  _QWORD *v41; // [rsp+10h] [rbp-150h]
  _QWORD *v42; // [rsp+10h] [rbp-150h]
  _QWORD *v43; // [rsp+18h] [rbp-148h]
  _QWORD *v44; // [rsp+20h] [rbp-140h]
  unsigned __int64 v45; // [rsp+20h] [rbp-140h]
  _QWORD *v46; // [rsp+20h] [rbp-140h]
  __int64 v48; // [rsp+38h] [rbp-128h]
  _QWORD v49[2]; // [rsp+40h] [rbp-120h] BYREF
  __int64 v50; // [rsp+50h] [rbp-110h] BYREF
  unsigned __int64 v51; // [rsp+58h] [rbp-108h]
  __int64 *v52; // [rsp+60h] [rbp-100h]
  __int64 v53; // [rsp+68h] [rbp-F8h]
  int v54; // [rsp+70h] [rbp-F0h]
  char v55; // [rsp+80h] [rbp-E0h]
  _QWORD v56[26]; // [rsp+90h] [rbp-D0h] BYREF

  if ( a1[26] < a1[27] )
  {
    v43 = a1 + 15;
    while ( 1 )
    {
      sub_C22680((__int64)&v50, (__int64)a1);
      if ( (v55 & 1) != 0 )
      {
        result = (unsigned int)v50;
        if ( (_DWORD)v50 )
          return result;
      }
      if ( v54 )
      {
        v4 = sub_C1B290(v52, &v52[3 * v53]);
      }
      else
      {
        v17 = v50;
        v4 = v51;
        if ( v50 )
        {
          sub_C7D030(v56);
          sub_C7D280(v56, v17, v4);
          sub_C7D290(v56, v49);
          v4 = v49[0];
        }
      }
      v5 = v4 % a1[2];
      v56[0] = v4;
      v6 = sub_C1DD00(a1 + 1, v5, v56, v4);
      if ( v6 && *v6 )
        v7 = *v6 + 16LL;
      else
        v7 = 0;
      v8 = a1[26];
      result = sub_C27850((__int64)a1, a2, v7);
      if ( (_DWORD)result )
        return result;
      v48 = a1[26];
      if ( v54 )
      {
        v9 = sub_C1B290(v52, &v52[3 * v53]);
      }
      else
      {
        v18 = v50;
        v9 = v51;
        if ( v50 )
        {
          sub_C7D030(v56);
          sub_C7D280(v56, v18, v9);
          sub_C7D290(v56, v49);
          v9 = v49[0];
        }
      }
      v10 = a1[14];
      v11 = *(_QWORD **)(a1[13] + 8 * (v9 % v10));
      v12 = v9 % v10;
      if ( !v11 )
        goto LABEL_27;
      v13 = (_QWORD *)*v11;
      if ( v9 != *(_QWORD *)(*v11 + 8LL) )
        break;
LABEL_18:
      v15 = *v11;
      v16 = (_QWORD *)(*v11 + 16LL);
      if ( !v15 )
        goto LABEL_27;
LABEL_19:
      *v16 = v8;
      v16[1] = v48;
      if ( a1[26] >= a1[27] )
        goto LABEL_20;
    }
    while ( 1 )
    {
      v14 = (_QWORD *)*v13;
      if ( !*v13 )
        break;
      v11 = v13;
      if ( v9 % v10 != v14[1] % v10 )
        break;
      v13 = (_QWORD *)*v13;
      if ( v9 == v14[1] )
        goto LABEL_18;
    }
LABEL_27:
    v19 = (_QWORD *)sub_22077B0(32);
    if ( v19 )
      *v19 = 0;
    v19[1] = v9;
    v20 = a1[14];
    v19[2] = 0;
    v21 = a1[16];
    v19[3] = 0;
    v44 = v19;
    v22 = sub_222DA10(a1 + 17, v20, v21, 1);
    v24 = v44;
    v25 = v23;
    if ( !v22 )
    {
      v26 = (_QWORD *)a1[13];
      v27 = (_QWORD **)&v26[v12];
      v28 = (_QWORD *)v26[v12];
      if ( v28 )
      {
LABEL_31:
        *v24 = *v28;
        **v27 = v24;
LABEL_32:
        ++a1[16];
        v16 = v24 + 2;
        goto LABEL_19;
      }
LABEL_46:
      v39 = a1[15];
      a1[15] = v24;
      *v24 = v39;
      if ( v39 )
      {
        v26[*(_QWORD *)(v39 + 8) % a1[14]] = v24;
        v27 = (_QWORD **)(v12 * 8 + a1[13]);
      }
      *v27 = v43;
      goto LABEL_32;
    }
    if ( v23 == 1 )
    {
      v26 = a1 + 19;
      a1[19] = 0;
      v32 = a1 + 19;
    }
    else
    {
      if ( v23 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 17, v20, v23, v23);
      v29 = 8 * v23;
      v41 = v44;
      v45 = v23;
      v30 = (void *)sub_22077B0(8 * v23);
      v31 = memset(v30, 0, v29);
      v24 = v41;
      v25 = v45;
      v32 = a1 + 19;
      v26 = v31;
    }
    v33 = (_QWORD *)a1[15];
    a1[15] = 0;
    if ( !v33 )
    {
LABEL_43:
      v38 = (_QWORD *)a1[13];
      if ( v38 != v32 )
      {
        v40 = v25;
        v42 = v24;
        v46 = v26;
        j_j___libc_free_0(v38, 8LL * a1[14]);
        v25 = v40;
        v24 = v42;
        v26 = v46;
      }
      a1[14] = v25;
      a1[13] = v26;
      v12 = v9 % v25;
      v27 = (_QWORD **)&v26[v12];
      v28 = (_QWORD *)v26[v12];
      if ( v28 )
        goto LABEL_31;
      goto LABEL_46;
    }
    v34 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v35 = v33;
        v33 = (_QWORD *)*v33;
        v36 = v35[1] % v25;
        v37 = (_QWORD **)&v26[v36];
        if ( !*v37 )
          break;
        *v35 = **v37;
        **v37 = v35;
LABEL_39:
        if ( !v33 )
          goto LABEL_43;
      }
      *v35 = a1[15];
      a1[15] = v35;
      *v37 = v43;
      if ( !*v35 )
      {
        v34 = v36;
        goto LABEL_39;
      }
      v26[v34] = v35;
      v34 = v36;
      if ( !v33 )
        goto LABEL_43;
    }
  }
LABEL_20:
  sub_C1AFD0();
  return 0;
}
