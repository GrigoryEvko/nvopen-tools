// Function: sub_26F78E0
// Address: 0x26f78e0
//
__int64 *__fastcall sub_26F78E0(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 *a7,
        size_t a8)
{
  __int64 v12; // rax
  size_t v13; // rdx
  _BYTE *v14; // rdi
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rax
  _QWORD *v17; // r8
  unsigned __int64 v18; // rsi
  unsigned __int64 *v19; // r14
  _QWORD *v20; // rdi
  unsigned __int64 v21; // r15
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  _QWORD *v24; // r14
  void *v25; // rdi
  __int64 v27; // rax
  size_t v28; // [rsp+10h] [rbp-80h]
  _QWORD v29[3]; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v30; // [rsp+38h] [rbp-58h]
  void *dest; // [rsp+40h] [rbp-50h]
  __int64 v32; // [rsp+48h] [rbp-48h]
  __int64 *v33; // [rsp+50h] [rbp-40h]

  *a1 = (__int64)(a1 + 2);
  sub_26F6410(a1, "__typeid_", (__int64)"");
  v29[1] = 0;
  v32 = 0x100000000LL;
  v29[2] = 0;
  v30 = 0;
  v29[0] = &unk_49DD210;
  dest = 0;
  v33 = a1;
  sub_CB5980((__int64)v29, 0, 0, 0);
  v12 = sub_B91420(a2);
  v14 = dest;
  v15 = (unsigned __int8 *)v12;
  v16 = v30;
  if ( v13 > v30 - (unsigned __int64)dest )
  {
    v27 = sub_CB6200((__int64)v29, v15, v13);
    v14 = *(_BYTE **)(v27 + 32);
    v17 = (_QWORD *)v27;
    v16 = *(_QWORD *)(v27 + 24);
  }
  else
  {
    v17 = v29;
    if ( v13 )
    {
      v28 = v13;
      memcpy(dest, v15, v13);
      v17 = v29;
      dest = (char *)dest + v28;
      v14 = dest;
      if ( v30 > (unsigned __int64)dest )
        goto LABEL_4;
      goto LABEL_18;
    }
  }
  if ( v16 > (unsigned __int64)v14 )
  {
LABEL_4:
    v17[4] = v14 + 1;
    *v14 = 95;
    goto LABEL_5;
  }
LABEL_18:
  v17 = (_QWORD *)sub_CB5D20((__int64)v17, 95);
LABEL_5:
  v18 = a3;
  v19 = &a4[a5];
  sub_CB59D0((__int64)v17, v18);
  while ( v19 != a4 )
  {
    v21 = *a4;
    v22 = dest;
    if ( (unsigned __int64)dest < v30 )
    {
      v20 = v29;
      dest = (char *)dest + 1;
      *v22 = 95;
    }
    else
    {
      v20 = (_QWORD *)sub_CB5D20((__int64)v29, 95);
    }
    ++a4;
    sub_CB59D0((__int64)v20, v21);
  }
  v23 = dest;
  if ( (unsigned __int64)dest >= v30 )
  {
    v24 = (_QWORD *)sub_CB5D20((__int64)v29, 95);
  }
  else
  {
    v24 = v29;
    dest = (char *)dest + 1;
    *v23 = 95;
  }
  v25 = (void *)v24[4];
  if ( a8 > v24[3] - (_QWORD)v25 )
  {
    sub_CB6200((__int64)v24, a7, a8);
  }
  else if ( a8 )
  {
    memcpy(v25, a7, a8);
    v24[4] += a8;
  }
  v29[0] = &unk_49DD210;
  sub_CB5840((__int64)v29);
  return a1;
}
