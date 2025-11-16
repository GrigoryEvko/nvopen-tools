// Function: sub_31DABF0
// Address: 0x31dabf0
//
void __fastcall sub_31DABF0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // r12
  void *v6; // r13
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r13
  int v9; // eax
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r12
  __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 (__fastcall *v14)(__int64); // rcx
  __int64 v15; // rax
  _QWORD **v16; // r13
  __int64 v17; // r12
  size_t v18; // rdx
  unsigned __int64 v19; // rbx
  char v20; // r14
  __int64 v21; // rax
  size_t v22; // rdx
  unsigned __int8 *v23; // rax
  __int64 v24; // rax
  unsigned int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  __int64 (__fastcall *v29)(__int64); // r9
  __int64 v30; // rdi
  unsigned __int64 v31; // rsi
  __int64 (__fastcall *v32)(__int64); // r8
  __int64 v33; // rsi
  __int64 v34; // rdi
  _BYTE *v35; // rax
  __int64 v36; // rdi
  _BYTE *v37; // rax
  __int64 v38; // rdi
  int v39; // r13d
  unsigned __int64 v40; // rsi
  __int64 (__fastcall *v41)(__int64); // r8
  unsigned int v42; // [rsp+Ch] [rbp-74h]
  unsigned __int8 *v43; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v44; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v46; // [rsp+30h] [rbp-50h] BYREF
  size_t v47; // [rsp+38h] [rbp-48h]
  __int64 v48; // [rsp+40h] [rbp-40h]
  _BYTE v49[56]; // [rsp+48h] [rbp-38h] BYREF

  v4 = (unsigned __int64 *)&v44;
  v6 = sub_C33340();
  if ( (void *)*a1 == v6 )
    sub_C3E660((__int64)&v44, (__int64)a1);
  else
    sub_C3A850((__int64)&v44, a1);
  if ( *(_BYTE *)(a3 + 488) )
  {
    v43 = v49;
    v46 = v49;
    v47 = 0;
    v48 = 8;
    if ( v6 == (void *)*a1 )
      sub_C40650((__int64)a1, (__int64 *)&v46, 0, 3u, 1u);
    else
      sub_C35AD0((__int64)a1, (__int64 *)&v46, 0, 3u, 1);
    v33 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 224) + 128LL))(*(_QWORD *)(a3 + 224));
    sub_A587F0(a2, v33, 0, 0);
    v34 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 224) + 128LL))(*(_QWORD *)(a3 + 224));
    v35 = *(_BYTE **)(v34 + 32);
    if ( (unsigned __int64)v35 >= *(_QWORD *)(v34 + 24) )
    {
      v34 = sub_CB5D20(v34, 32);
    }
    else
    {
      *(_QWORD *)(v34 + 32) = v35 + 1;
      *v35 = 32;
    }
    v36 = sub_CB6200(v34, v46, v47);
    v37 = *(_BYTE **)(v36 + 32);
    if ( (unsigned __int64)v37 >= *(_QWORD *)(v36 + 24) )
    {
      sub_CB5D20(v36, 10);
    }
    else
    {
      *(_QWORD *)(v36 + 32) = v37 + 1;
      *v37 = 10;
    }
    if ( v46 != v43 )
      _libc_free((unsigned __int64)v46);
  }
  v7 = v45 >> 3;
  if ( v45 > 0x40 )
    v4 = v44;
  v42 = (v45 >> 3) & 7;
  if ( !*(_BYTE *)sub_31DA930(a3) || *(_BYTE *)(a2 + 8) == 6 )
  {
    v43 = (unsigned __int8 *)(v7 >> 3);
    if ( v7 >> 3 )
    {
      v25 = 0;
      v26 = 0;
      do
      {
        while ( 1 )
        {
          v27 = *(_QWORD *)(a3 + 224);
          v28 = v4[v26];
          v29 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 560LL);
          if ( v29 != sub_C13FC0 )
            break;
          (*(void (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v27 + 536LL))(v27, v28, 8);
          v26 = ++v25;
          if ( v25 >= (unsigned __int64)v43 )
            goto LABEL_25;
        }
        ((void (__fastcall *)(__int64, unsigned __int64, __int64))v29)(v27, v28, 8);
        v26 = ++v25;
      }
      while ( v25 < (unsigned __int64)v43 );
    }
    else
    {
      v26 = 0;
    }
LABEL_25:
    if ( v42 )
    {
      v30 = *(_QWORD *)(a3 + 224);
      v31 = v4[v26];
      v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 560LL);
      if ( v32 == sub_C13FC0 )
        (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v30 + 536LL))(v30, v31);
      else
        ((void (__fastcall *)(__int64, unsigned __int64, _QWORD))v32)(v30, v31, v42);
    }
  }
  else
  {
    v8 = ((unsigned __int64)v45 + 63) >> 6;
    v9 = v8 - 1;
    if ( v42 )
    {
      v38 = *(_QWORD *)(a3 + 224);
      v39 = v8 - 2;
      v40 = v4[v9];
      v41 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v38 + 560LL);
      if ( v41 == sub_C13FC0 )
        (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v38 + 536LL))(v38, v40);
      else
        ((void (__fastcall *)(__int64, unsigned __int64, _QWORD))v41)(v38, v40, v42);
      v9 = v39;
    }
    if ( v9 >= 0 )
    {
      v10 = &v4[v9];
      v11 = v4 - 1;
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(a3 + 224);
          v13 = *v10;
          v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 560LL);
          if ( v14 != sub_C13FC0 )
            break;
          (*(void (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v12 + 536LL))(v12, v13, 8);
          if ( v11 == --v10 )
            goto LABEL_14;
        }
        ((void (__fastcall *)(__int64, unsigned __int64, __int64))v14)(v12, v13, 8);
        --v10;
      }
      while ( v11 != v10 );
    }
  }
LABEL_14:
  v15 = sub_31DA930(a3);
  v16 = *(_QWORD ***)(a3 + 224);
  v17 = v15;
  v46 = (unsigned __int8 *)sub_9208B0(v15, a2);
  v47 = v18;
  v19 = (unsigned __int64)(v46 + 7) >> 3;
  LOBYTE(v43) = v18;
  v20 = sub_AE5020(v17, a2);
  v21 = sub_9208B0(v17, a2);
  v47 = v22;
  v23 = (unsigned __int8 *)((((1LL << v20) + ((unsigned __int64)(v21 + 7) >> 3) - 1) >> v20 << v20) - v19);
  if ( v19 )
    LOBYTE(v22) = (_BYTE)v43;
  v46 = v23;
  LOBYTE(v47) = v22;
  v24 = sub_CA1930(&v46);
  sub_E99300(v16, v24);
  if ( v45 > 0x40 )
  {
    if ( v44 )
      j_j___libc_free_0_0((unsigned __int64)v44);
  }
}
