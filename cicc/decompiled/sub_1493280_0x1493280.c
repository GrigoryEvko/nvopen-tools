// Function: sub_1493280
// Address: 0x1493280
//
__int64 __fastcall sub_1493280(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7,
        __m128i a8,
        bool a9,
        char a10)
{
  char v10; // r11
  __int64 v13; // rbx
  bool v14; // zf
  bool v15; // r9
  bool v16; // r8
  __int64 v17; // rax
  __int64 *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  char v27; // al
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  char v31; // cl
  bool v32; // al
  char v33; // cl
  bool v34; // al
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rax
  char v38; // [rsp+0h] [rbp-C0h]
  char v39; // [rsp+10h] [rbp-B0h]
  __int64 v40; // [rsp+10h] [rbp-B0h]
  __int64 v41; // [rsp+18h] [rbp-A8h]
  char v44; // [rsp+2Ch] [rbp-94h]
  __int64 v45[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v46; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v47; // [rsp+48h] [rbp-78h]
  _BYTE *v48; // [rsp+50h] [rbp-70h]
  __int64 v49; // [rsp+58h] [rbp-68h]
  int v50; // [rsp+60h] [rbp-60h]
  _BYTE v51[88]; // [rsp+68h] [rbp-58h] BYREF

  v10 = 0;
  v13 = a3;
  v14 = *(_WORD *)(a3 + 24) == 7;
  v15 = a9;
  v46 = 0;
  v47 = v51;
  v16 = a9;
  v48 = v51;
  v49 = 4;
  v50 = 0;
  if ( v14 )
  {
    if ( a5 != *(_QWORD *)(a3 + 48) )
    {
LABEL_3:
      v17 = sub_1456E90((__int64)a2);
      sub_14573F0(a1, v17);
      goto LABEL_4;
    }
  }
  else
  {
    if ( !a10 )
      goto LABEL_3;
    v19 = sub_1493080((__int64)a2, a3, a5, (__int64)&v46, a7, a8);
    v16 = a9;
    v15 = a9;
    v10 = a10;
    v13 = (__int64)v19;
    if ( !v19 || a5 != v19[6] )
      goto LABEL_3;
  }
  if ( *(_QWORD *)(v13 + 40) != 2 )
    goto LABEL_3;
  if ( v15 )
    v16 = ((a6 == 0 ? 2 : 4) & *(unsigned __int16 *)(v13 + 26)) != 0;
  v39 = v16;
  v38 = v10;
  v41 = sub_13A5BC0((_QWORD *)v13, (__int64)a2);
  if ( (unsigned __int8)sub_1477C30((__int64)a2, v41) )
  {
    if ( !sub_1456110(v41) && (unsigned __int8)sub_1481FB0((__int64)a2, a4, v41, a6, v39) )
      goto LABEL_3;
  }
  else if ( !byte_4F9AA80
         || v39 != 1
         || v38
         || (unsigned __int8)sub_1477A90((__int64)a2, v41)
         || !((unsigned __int16)sub_14691E0((__int64)a2, a5) >> 8) )
  {
    goto LABEL_3;
  }
  v20 = **(_QWORD **)(v13 + 32);
  if ( sub_146CEE0((__int64)a2, a4, a5) )
  {
    v25 = sub_14806B0((__int64)a2, a4, v20, 0, 0);
    v40 = sub_1484BE0(a2, v25, v41, 0, a7, a8);
    v26 = sub_14806B0((__int64)a2, v20, v41, 0, 0);
    v27 = sub_148B410((__int64)a2, a5, a6 == 0 ? 36 : 40, v26, a4);
    v28 = v40;
    if ( !v27 )
    {
      if ( a6 )
        v29 = sub_147A9C0(a2, a4, v20, a7, a8);
      else
        v29 = sub_14819D0(a2, a4, v20, a7, a8);
      v30 = sub_14806B0((__int64)a2, v29, v20, 0, 0);
      v28 = sub_1484BE0(a2, v30, v41, 0, a7, a8);
    }
    if ( *(_WORD *)(v28 + 24) )
    {
      v31 = 1;
      if ( *(_WORD *)(v40 + 24) )
      {
        v35 = sub_1456040(a3);
        v36 = sub_1456C90((__int64)a2, v35);
        v37 = sub_1484C70(a2, v20, v41, a4, v36, a6, a7, a8);
        v31 = 0;
        v40 = v37;
      }
    }
    else
    {
      v40 = v28;
      v31 = 0;
    }
    v44 = v31;
    v32 = sub_14562D0(v40);
    v33 = v44;
    if ( v32 )
    {
      v34 = sub_14562D0(v28);
      v33 = v44;
      if ( !v34 )
      {
        sub_1477A60((__int64)v45, (__int64)a2, v28);
        v40 = sub_145CF40((__int64)a2, (__int64)v45);
        sub_135E100(v45);
        v33 = v44;
      }
    }
    sub_14575B0(a1, v28, v40, v33, (__int64)&v46);
  }
  else
  {
    v21 = sub_1456040(a3);
    v22 = sub_1456C90((__int64)a2, v21);
    v23 = sub_1484C70(a2, v20, v41, a4, v22, a6, a7, a8);
    v24 = sub_1456E90((__int64)a2);
    sub_14575B0(a1, v24, v23, 0, (__int64)&v46);
  }
LABEL_4:
  if ( v48 != v47 )
    _libc_free((unsigned __int64)v48);
  return a1;
}
