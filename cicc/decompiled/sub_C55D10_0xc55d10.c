// Function: sub_C55D10
// Address: 0xc55d10
//
__int64 __fastcall sub_C55D10(
        unsigned __int8 (__fastcall ***a1)(_QWORD, _QWORD),
        __int64 a2,
        unsigned __int8 (__fastcall ***a3)(_QWORD, __int64),
        unsigned __int8 (__fastcall ***a4)(_QWORD, __int64),
        int a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int8 (__fastcall ***v15)(_QWORD, _QWORD); // rdi
  unsigned int v16; // r13d
  unsigned __int8 (__fastcall *v17)(_QWORD, __int64); // r15
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v21; // rax
  __int64 v22; // rbx
  const void *v23; // rax
  size_t v24; // rdx
  void *v25; // rdi
  __int64 v26; // rsi
  unsigned int v27; // r13d
  unsigned __int64 v28; // rdx
  unsigned int v29; // ebx
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int8 (__fastcall *v32)(_QWORD, __int64); // r13
  __int64 v33; // rax
  void *v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // r13
  const void *v37; // rax
  size_t v38; // rdx
  size_t v39; // [rsp+0h] [rbp-60h]
  int v40; // [rsp+8h] [rbp-58h]
  size_t v41; // [rsp+8h] [rbp-58h]
  _QWORD v42[10]; // [rsp+10h] [rbp-50h] BYREF

  v9 = sub_CB7210(a1);
  v10 = sub_904010(v9, "  ");
  v11 = *(_QWORD *)(a2 + 24);
  v42[2] = 2;
  v12 = v10;
  v13 = *(_QWORD *)(a2 + 32);
  v42[0] = v11;
  v42[1] = v13;
  sub_C51AE0(v12, (__int64)v42);
  v14 = sub_CB7210(v12);
  sub_CB69B0(v14, (unsigned int)(a5 - *(_DWORD *)(a2 + 32)));
  v15 = a1;
  v40 = ((__int64 (__fastcall *)(unsigned __int8 (__fastcall ***)(_QWORD, _QWORD)))(*a1)[2])(a1);
  if ( v40 )
  {
    v16 = 0;
    while ( 1 )
    {
      v17 = **a3;
      v18 = ((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[6])(a1, v16);
      v15 = a3;
      if ( v17(a3, v18) )
        break;
      if ( v40 == ++v16 )
        goto LABEL_5;
    }
    v21 = sub_CB7210(a3);
    v22 = sub_904010(v21, "= ");
    v23 = (const void *)((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[3])(a1, v16);
    v25 = *(void **)(v22 + 32);
    if ( v24 > *(_QWORD *)(v22 + 24) - (_QWORD)v25 )
    {
      sub_CB6200(v22, v23, v24);
    }
    else if ( v24 )
    {
      v39 = v24;
      memcpy(v25, v23, v24);
      *(_QWORD *)(v22 + 32) += v39;
    }
    v26 = v16;
    v27 = 0;
    (*a1)[3](a1, v26);
    if ( v28 < 8 )
      v27 = 8 - v28;
    v29 = 0;
    v30 = sub_CB7210(a1);
    v31 = sub_CB69B0(v30, v27);
    sub_904010(v31, " (default: ");
    while ( 1 )
    {
      v32 = **a4;
      v33 = ((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[6])(a1, v29);
      v34 = a4;
      if ( v32(a4, v33) )
        break;
      if ( v40 == ++v29 )
        goto LABEL_14;
    }
    v36 = sub_CB7210(a4);
    v37 = (const void *)((__int64 (__fastcall *)(_QWORD, _QWORD))(*a1)[3])(a1, v29);
    v34 = *(void **)(v36 + 32);
    if ( v38 > *(_QWORD *)(v36 + 24) - (_QWORD)v34 )
    {
      v34 = (void *)v36;
      sub_CB6200(v36, v37, v38);
    }
    else if ( v38 )
    {
      v41 = v38;
      memcpy(v34, v37, v38);
      *(_QWORD *)(v36 + 32) += v41;
    }
LABEL_14:
    v35 = sub_CB7210(v34);
    return sub_904010(v35, ")\n");
  }
  else
  {
LABEL_5:
    v19 = sub_CB7210(v15);
    return sub_904010(v19, "= *unknown option value*\n");
  }
}
