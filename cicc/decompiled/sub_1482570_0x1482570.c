// Function: sub_1482570
// Address: 0x1482570
//
__int64 __fastcall sub_1482570(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6, __m128i a7)
{
  char v9; // al
  unsigned int v10; // r14d
  __int64 result; // rax
  int v12; // eax
  __int64 *v13; // r8
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r15
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-40h]
  __int64 v46; // [rsp+0h] [rbp-40h]
  __int64 v47; // [rsp+0h] [rbp-40h]
  __int64 v48; // [rsp+0h] [rbp-40h]
  __int64 v49; // [rsp+0h] [rbp-40h]
  __int64 v50; // [rsp+0h] [rbp-40h]
  __int64 v51; // [rsp+0h] [rbp-40h]
  __int64 v52; // [rsp+0h] [rbp-40h]
  __int64 v53; // [rsp+0h] [rbp-40h]
  __int64 v54; // [rsp+0h] [rbp-40h]
  unsigned __int64 v55; // [rsp+8h] [rbp-38h]
  __int64 v56; // [rsp+8h] [rbp-38h]
  __int64 v57; // [rsp+8h] [rbp-38h]
  unsigned __int64 v58; // [rsp+8h] [rbp-38h]
  __int64 v59; // [rsp+8h] [rbp-38h]
  __int64 v60; // [rsp+8h] [rbp-38h]
  unsigned __int64 v61; // [rsp+8h] [rbp-38h]
  __int64 v62; // [rsp+8h] [rbp-38h]
  unsigned __int64 v63; // [rsp+8h] [rbp-38h]
  __int64 v64; // [rsp+8h] [rbp-38h]
  __int64 v65; // [rsp+8h] [rbp-38h]

  v9 = *(_BYTE *)(a3 + 16);
  if ( v9 == 13 )
  {
    v10 = *(_DWORD *)(a3 + 32);
    if ( v10 <= 0x40 )
    {
      if ( *(_QWORD *)(a3 + 24) == 1 )
        a5 = a4;
    }
    else if ( (unsigned int)sub_16A57B0(a3 + 24) == v10 - 1 )
    {
      a5 = a4;
    }
    return sub_146F1B0((__int64)a1, a5);
  }
  else if ( v9 == 75 )
  {
    v12 = *(unsigned __int16 *)(a3 + 18);
    v13 = *(__int64 **)(a3 - 48);
    v14 = *(_QWORD *)(a3 - 24);
    BYTE1(v12) &= ~0x80u;
    switch ( v12 )
    {
      case ' ':
        v51 = *(_QWORD *)(a3 - 48);
        v61 = sub_1456C90((__int64)a1, *v13);
        if ( v61 > sub_1456C90((__int64)a1, *a2) || *(_BYTE *)(v14 + 16) != 13 || !sub_13D01C0(v14 + 24) )
          return sub_145DC80((__int64)a1, (__int64)a2);
        v31 = sub_145CF80((__int64)a1, *a2, 1, 0);
        v32 = *a2;
        v62 = v31;
        v33 = sub_146F1B0((__int64)a1, v51);
        v34 = sub_14758B0((__int64)a1, v33, v32);
        v52 = sub_146F1B0((__int64)a1, a4);
        v35 = sub_146F1B0((__int64)a1, a5);
        v36 = sub_14806B0((__int64)a1, v52, v62, 0, 0);
        v37 = v34;
        v38 = v36;
        goto LABEL_27;
      case '!':
        v53 = *(_QWORD *)(a3 - 48);
        v63 = sub_1456C90((__int64)a1, *v13);
        if ( v63 > sub_1456C90((__int64)a1, *a2) )
          return sub_145DC80((__int64)a1, (__int64)a2);
        if ( *(_BYTE *)(v14 + 16) != 13 )
          return sub_145DC80((__int64)a1, (__int64)a2);
        v64 = v53;
        if ( !sub_13D01C0(v14 + 24) )
          return sub_145DC80((__int64)a1, (__int64)a2);
        v40 = sub_145CF80((__int64)a1, *a2, 1, 0);
        v41 = *a2;
        v54 = v40;
        v42 = sub_146F1B0((__int64)a1, v64);
        v34 = sub_14758B0((__int64)a1, v42, v41);
        v65 = sub_146F1B0((__int64)a1, a4);
        v35 = sub_146F1B0((__int64)a1, a5);
        v38 = sub_14806B0((__int64)a1, v65, v34, 0, 0);
        v62 = v54;
        v37 = v54;
LABEL_27:
        if ( v38 != sub_14806B0((__int64)a1, v35, v37, 0, 0) )
          return sub_145DC80((__int64)a1, (__int64)a2);
        v39 = sub_14819D0(a1, v62, v34, a6, a7);
        v22 = v38;
        v23 = v39;
        goto LABEL_17;
      case '"':
      case '#':
        goto LABEL_19;
      case '$':
      case '%':
        v13 = *(__int64 **)(a3 - 24);
        v14 = *(_QWORD *)(a3 - 48);
LABEL_19:
        v48 = (__int64)v13;
        v58 = sub_1456C90((__int64)a1, *v13);
        if ( v58 > sub_1456C90((__int64)a1, *a2) )
          return sub_145DC80((__int64)a1, (__int64)a2);
        v59 = *a2;
        v24 = sub_146F1B0((__int64)a1, v48);
        v60 = sub_14758B0((__int64)a1, v24, v59);
        v49 = *a2;
        v25 = sub_146F1B0((__int64)a1, v14);
        v26 = sub_14758B0((__int64)a1, v25, v49);
        v27 = sub_146F1B0((__int64)a1, a4);
        v28 = sub_146F1B0((__int64)a1, a5);
        v50 = sub_14806B0((__int64)a1, v27, v60, 0, 0);
        if ( v50 == sub_14806B0((__int64)a1, v28, v26, 0, 0) )
        {
          v43 = sub_14819D0(a1, v60, v26, a6, a7);
          v22 = v50;
          v23 = v43;
        }
        else
        {
          v29 = sub_14806B0((__int64)a1, v27, v26, 0, 0);
          if ( v29 != sub_14806B0((__int64)a1, v28, v60, 0, 0) )
            return sub_145DC80((__int64)a1, (__int64)a2);
          v30 = sub_1481BD0(a1, v60, v26, a6, a7);
          v22 = v29;
          v23 = v30;
        }
        goto LABEL_17;
      case '&':
      case '\'':
        goto LABEL_13;
      case '(':
      case ')':
        v13 = *(__int64 **)(a3 - 24);
        v14 = *(_QWORD *)(a3 - 48);
LABEL_13:
        v45 = (__int64)v13;
        v55 = sub_1456C90((__int64)a1, *v13);
        if ( v55 > sub_1456C90((__int64)a1, *a2) )
          return sub_145DC80((__int64)a1, (__int64)a2);
        v56 = *a2;
        v15 = sub_146F1B0((__int64)a1, v45);
        v57 = sub_147BE00((__int64)a1, v15, v56);
        v46 = *a2;
        v16 = sub_146F1B0((__int64)a1, v14);
        v17 = sub_147BE00((__int64)a1, v16, v46);
        v18 = sub_146F1B0((__int64)a1, a4);
        v19 = sub_146F1B0((__int64)a1, a5);
        v47 = sub_14806B0((__int64)a1, v18, v57, 0, 0);
        if ( v47 == sub_14806B0((__int64)a1, v19, v17, 0, 0) )
        {
          v44 = sub_147A9C0(a1, v57, v17, a6, a7);
          v22 = v47;
          v23 = v44;
        }
        else
        {
          v20 = sub_14806B0((__int64)a1, v18, v17, 0, 0);
          if ( v20 != sub_14806B0((__int64)a1, v19, v57, 0, 0) )
            return sub_145DC80((__int64)a1, (__int64)a2);
          v21 = sub_1480950(a1, v57, v17, a6, a7);
          v22 = v20;
          v23 = v21;
        }
LABEL_17:
        result = sub_13A5B00((__int64)a1, v23, v22, 0, 0);
        break;
      default:
        return sub_145DC80((__int64)a1, (__int64)a2);
    }
  }
  else
  {
    return sub_145DC80((__int64)a1, (__int64)a2);
  }
  return result;
}
