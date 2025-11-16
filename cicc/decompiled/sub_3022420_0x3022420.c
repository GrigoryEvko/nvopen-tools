// Function: sub_3022420
// Address: 0x3022420
//
__int64 __fastcall sub_3022420(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 result; // rax
  _WORD *v10; // rdx
  unsigned __int8 v11; // al
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // ebx
  char v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  _WORD *v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int); // rax
  int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  __int64 v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+10h] [rbp-40h] BYREF
  __int64 v34; // [rsp+18h] [rbp-38h]

  v6 = sub_31DA930();
  v7 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 200) + 16LL))(*(_QWORD *)(a1 + 200), a3);
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 144LL);
  if ( v8 == sub_3020010 )
    v32 = v7 + 960;
  else
    v32 = v8(v7);
  result = *(unsigned __int8 *)(a2 + 8);
  if ( (_DWORD)result == 15 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 1) == 0 )
      return result;
  }
  else if ( (_DWORD)result == 7 )
  {
    return result;
  }
  v10 = *(_WORD **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v10 <= 1u )
  {
    sub_CB6200(a4, (unsigned __int8 *)" (", 2u);
    v11 = *(_BYTE *)(a2 + 8);
    if ( v11 <= 3u )
      goto LABEL_7;
  }
  else
  {
    *v10 = 10272;
    *(_QWORD *)(a4 + 32) += 2LL;
    v11 = *(_BYTE *)(a2 + 8);
    if ( v11 <= 3u )
      goto LABEL_7;
  }
  if ( v11 == 5 || (v11 & 0xFD) == 4 )
  {
LABEL_7:
    if ( (unsigned __int8)sub_30201A0(a2) )
    {
LABEL_8:
      v11 = *(_BYTE *)(a2 + 8);
      goto LABEL_9;
    }
LABEL_23:
    if ( *(_BYTE *)(a2 + 8) == 12 )
    {
      v21 = *(_DWORD *)(a2 + 8) >> 8;
    }
    else
    {
      v33 = sub_BCAE30(a2);
      v34 = v31;
      v21 = sub_CA1930(&v33);
    }
    v22 = 32;
    if ( v21 > 0x20 )
    {
      v22 = 64;
      if ( v21 >= 0x40 )
        v22 = v21;
    }
    v23 = sub_904010(a4, ".param .b");
    v24 = sub_CB59D0(v23, v22);
    sub_904010(v24, " func_retval0");
    goto LABEL_12;
  }
  if ( v11 == 12 )
  {
    if ( (unsigned __int8)sub_30201A0(a2) )
      goto LABEL_8;
    goto LABEL_23;
  }
LABEL_9:
  if ( v11 == 14 )
  {
    v25 = sub_904010(a4, ".param .b");
    v26 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v32 + 32LL);
    if ( v26 == sub_2D42F30 )
    {
      v27 = sub_AE2980(v6, 0)[1];
      switch ( v27 )
      {
        case 1:
          v28 = 2;
          goto LABEL_40;
        case 2:
          v28 = 3;
          goto LABEL_40;
        case 4:
          v28 = 4;
          goto LABEL_40;
        case 8:
          v28 = 5;
          goto LABEL_40;
        case 16:
          v28 = 6;
          goto LABEL_40;
        case 32:
          v28 = 7;
          goto LABEL_40;
        case 64:
          v28 = 8;
          goto LABEL_40;
        case 128:
          v28 = 9;
LABEL_40:
          v29 = 16LL * (v28 - 1) + 71615648;
          v30 = *(_QWORD *)v29;
          if ( *(_BYTE *)(v29 + 8) )
            sub_904010(v25, "vscale x ");
          sub_CB59D0(v25, v30);
          sub_904010(v25, " func_retval0");
          goto LABEL_12;
      }
    }
    else
    {
      v28 = (unsigned __int16)v26(v32, v6, 0);
      if ( (unsigned __int16)v28 > 1u && (unsigned __int16)(v28 - 504) > 7u )
        goto LABEL_40;
    }
LABEL_54:
    BUG();
  }
  if ( !(unsigned __int8)sub_30201A0(a2) )
    goto LABEL_54;
  v12 = sub_BDB740(v6, a2);
  v34 = v13;
  v33 = v12;
  v14 = sub_CA1930(&v33);
  v15 = sub_303E6A0(v32, a3, a2, 0, v6);
  v16 = sub_904010(a4, ".param .align ");
  v17 = sub_CB59D0(v16, 1LL << v15);
  v18 = sub_904010(v17, " .b8 func_retval0[");
  v19 = sub_CB59D0(v18, v14);
  sub_904010(v19, "]");
LABEL_12:
  v20 = *(_WORD **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v20 <= 1u )
    return sub_CB6200(a4, (unsigned __int8 *)") ", 2u);
  result = 8233;
  *v20 = 8233;
  *(_QWORD *)(a4 + 32) += 2LL;
  return result;
}
