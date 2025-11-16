// Function: sub_1494580
// Address: 0x1494580
//
__int64 __fastcall sub_1494580(
        __int64 a1,
        _QWORD *a2,
        int a3,
        _QWORD *a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7,
        __m128i a8,
        unsigned __int8 a9,
        char a10)
{
  unsigned __int8 v14; // cl
  unsigned __int8 v15; // r10
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rsi
  bool v23; // al
  unsigned __int64 v24; // rdi
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rbx
  __int64 v29; // rbx
  __int64 v30; // rbx
  __int64 v31; // rbx
  __int64 v32; // rbx
  __int64 v33; // rbx
  __int64 v34; // rbx
  char v35; // [rsp+10h] [rbp-110h]
  char v36; // [rsp+10h] [rbp-110h]
  int v37; // [rsp+14h] [rbp-10Ch]
  int v39; // [rsp+14h] [rbp-10Ch]
  char *v42; // [rsp+20h] [rbp-100h] BYREF
  _BYTE *v43; // [rsp+28h] [rbp-F8h]
  __int64 v44; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v45; // [rsp+38h] [rbp-E8h]
  char v46; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v47; // [rsp+50h] [rbp-D0h]
  unsigned __int64 v48; // [rsp+58h] [rbp-C8h]
  __int64 v49; // [rsp+90h] [rbp-90h] BYREF
  __int64 v50; // [rsp+98h] [rbp-88h]
  char v51; // [rsp+A0h] [rbp-80h]
  _BYTE v52[8]; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v53; // [rsp+B0h] [rbp-70h]
  unsigned __int64 v54; // [rsp+B8h] [rbp-68h]

  v14 = *(_BYTE *)(a5 + 16);
  if ( v14 <= 0x17u )
  {
    v15 = a6;
    if ( v14 == 13 )
    {
      v16 = *(_QWORD **)(a5 + 24);
      if ( *(_DWORD *)(a5 + 32) > 0x40u )
        v16 = (_QWORD *)*v16;
      if ( (v16 == 0) == a6 )
        v17 = sub_1456E90((__int64)a2);
      else
        v17 = sub_145CF80((__int64)a2, *(_QWORD *)a5, 0, 0);
      sub_14573F0(a1, v17);
      return a1;
    }
LABEL_10:
    v19 = sub_146AAD0((__int64)a2, (__int64)a4, a5, v15);
    sub_14573F0(a1, v19);
    return a1;
  }
  v15 = a6;
  if ( (unsigned int)v14 - 35 > 0x11 )
    goto LABEL_9;
  if ( v14 != 50 )
  {
    if ( v14 == 51 )
    {
      v35 = (a6 ^ 1) & a9;
      v37 = a6;
      sub_1494B40((unsigned int)&v44, (_DWORD)a2, a3, (_DWORD)a4, *(_QWORD *)(a5 - 48), a6, v35, a10);
      sub_1494B40((unsigned int)&v49, (_DWORD)a2, a3, (_DWORD)a4, *(_QWORD *)(a5 - 24), v37, v35, a10);
      v20 = sub_1456E90((__int64)a2);
      v21 = sub_1456E90((__int64)a2);
      if ( a6 )
      {
        v27 = v44;
        if ( v27 == sub_1456E90((__int64)a2) || (v28 = v49, v28 == sub_1456E90((__int64)a2)) )
          v20 = sub_1456E90((__int64)a2);
        else
          v20 = sub_1481DB0(a2, v44, v49, a7, a8);
        v29 = v45;
        if ( v29 == sub_1456E90((__int64)a2) )
        {
          v21 = v50;
        }
        else
        {
          v30 = v50;
          if ( v30 == sub_1456E90((__int64)a2) )
            v21 = v45;
          else
            v21 = sub_1481DB0(a2, v45, v50, a7, a8);
        }
      }
      else
      {
        if ( v45 == v50 )
          v21 = v50;
        if ( v44 == v49 )
          v20 = v49;
      }
      v42 = &v46;
      v22 = v20;
      v43 = v52;
      goto LABEL_32;
    }
LABEL_9:
    if ( v14 != 75 )
      goto LABEL_10;
    sub_1493EB0((__int64)&v49, a2, a4, a5, a6, a9, a7, a8, 0);
    v23 = sub_14562D0(v49);
    if ( a10 && v23 )
    {
      sub_1493EB0(a1, a2, a4, a5, a6, a9, a7, a8, 1);
    }
    else
    {
      *(_QWORD *)a1 = v49;
      *(_QWORD *)(a1 + 8) = v50;
      *(_BYTE *)(a1 + 16) = v51;
      sub_16CCEE0(a1 + 24, a1 + 64, 4, v52);
    }
    v24 = v54;
    if ( v54 == v53 )
      return a1;
LABEL_24:
    _libc_free(v24);
    return a1;
  }
  v36 = a6 & a9;
  v39 = a6;
  sub_1494B40((unsigned int)&v44, (_DWORD)a2, a3, (_DWORD)a4, *(_QWORD *)(a5 - 48), a6, a6 & a9, a10);
  sub_1494B40((unsigned int)&v49, (_DWORD)a2, a3, (_DWORD)a4, *(_QWORD *)(a5 - 24), v39, v36, a10);
  v25 = sub_1456E90((__int64)a2);
  v26 = sub_1456E90((__int64)a2);
  if ( a6 )
  {
    if ( v45 == v50 )
      v26 = v50;
    if ( v44 == v49 )
      v25 = v49;
  }
  else
  {
    v31 = v44;
    if ( v31 == sub_1456E90((__int64)a2) || (v32 = v49, v32 == sub_1456E90((__int64)a2)) )
      v25 = sub_1456E90((__int64)a2);
    else
      v25 = sub_1481DB0(a2, v44, v49, a7, a8);
    v33 = v45;
    if ( v33 == sub_1456E90((__int64)a2) )
    {
      v26 = v50;
    }
    else
    {
      v34 = v50;
      if ( v34 == sub_1456E90((__int64)a2) )
        v26 = v45;
      else
        v26 = sub_1481DB0(a2, v45, v50, a7, a8);
    }
  }
  if ( sub_14562D0(v26) && !sub_14562D0(v25) )
  {
    sub_1477A60((__int64)&v42, (__int64)a2, v25);
    v26 = sub_145CF40((__int64)a2, (__int64)&v42);
    sub_135E100((__int64 *)&v42);
  }
  v21 = v26;
  v42 = &v46;
  v22 = v25;
  v43 = v52;
LABEL_32:
  sub_1457420(a1, v22, v21, 0, &v42, 2);
  if ( v54 != v53 )
    _libc_free(v54);
  v24 = v48;
  if ( v48 != v47 )
    goto LABEL_24;
  return a1;
}
