// Function: sub_DD9390
// Address: 0xdd9390
//
_QWORD *__fastcall sub_DD9390(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, _BYTE *a5)
{
  __int64 v5; // r14
  __int64 v6; // r15
  unsigned __int64 v9; // rbx
  unsigned __int64 v11; // rbx
  unsigned int v12; // ebx
  __int64 v13; // rdi
  int v14; // eax
  bool v15; // al
  __int64 *v16; // rax
  _QWORD *v17; // rbx
  __int64 *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // r11
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  bool v26; // al
  __int64 *i; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rbx
  __int64 *v30; // r14
  _QWORD *v31; // rax
  __int64 *v33; // rbx
  __int64 v34; // r14
  __int64 v35; // r15
  __int64 v36; // rcx
  _QWORD *v37; // r13
  __int64 v38; // rcx
  __int64 v39; // rcx
  __int64 v40; // rsi
  int v41; // eax
  _QWORD *v42; // [rsp+8h] [rbp-78h]
  _QWORD *v43; // [rsp+10h] [rbp-70h]
  __int64 *v45; // [rsp+18h] [rbp-68h]
  unsigned int v46; // [rsp+18h] [rbp-68h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  bool v48; // [rsp+18h] [rbp-68h]
  __int64 v49; // [rsp+20h] [rbp-60h]
  _BYTE *v50; // [rsp+28h] [rbp-58h]
  __int64 *v51; // [rsp+28h] [rbp-58h]
  __int64 v52; // [rsp+40h] [rbp-40h]

  v5 = *(_QWORD *)(a3 - 64);
  v50 = a4;
  v49 = (__int64)a5;
  v6 = *(_QWORD *)(a3 - 32);
  switch ( *(_WORD *)(a3 + 2) & 0x3F )
  {
    case ' ':
      goto LABEL_6;
    case '!':
      v49 = (__int64)a4;
      v50 = a5;
LABEL_6:
      v11 = sub_D97050((__int64)a1, *(_QWORD *)(v5 + 8));
      if ( v11 > sub_D97050((__int64)a1, a2) )
        goto LABEL_16;
      if ( *(_BYTE *)v6 != 17 )
        return (_QWORD *)v52;
      v12 = *(_DWORD *)(v6 + 32);
      v13 = v6 + 24;
      if ( v12 <= 0x40 )
      {
        v15 = *(_QWORD *)(v6 + 24) == 0;
      }
      else
      {
        v14 = sub_C444A0(v13);
        v13 = v6 + 24;
        v15 = v12 == v14;
      }
      if ( !v15 )
        goto LABEL_18;
      v16 = sub_DD8400((__int64)a1, v5);
      v17 = sub_DC2CB0((__int64)a1, (__int64)v16, a2);
      v45 = sub_DD8400((__int64)a1, (__int64)v50);
      v18 = sub_DD8400((__int64)a1, v49);
      v43 = sub_DCC810(a1, (__int64)v18, (__int64)v17, 0, 0);
      v19 = sub_DCC810(a1, (__int64)v45, (__int64)v43, 0, 0);
      v20 = (__int64)v19;
      if ( *((_WORD *)v19 + 12) )
        goto LABEL_16;
      v21 = v19[4];
      v22 = v43;
      v23 = *(unsigned int *)(v21 + 32);
      v46 = *(_DWORD *)(v21 + 32);
      if ( (unsigned int)v23 > 0x40 )
      {
        v42 = v19;
        v41 = sub_C444A0(v21 + 24);
        v23 = v46;
        if ( v46 - v41 > 0x40 )
          goto LABEL_16;
        v20 = (__int64)v42;
        v22 = v43;
        v24 = **(_QWORD **)(v21 + 24);
      }
      else
      {
        v24 = *(_QWORD *)(v21 + 24);
      }
      v47 = (__int64)v22;
      if ( v24 <= 1 )
      {
        v25 = sub_DCE050(a1, (__int64)v17, v20, v23);
        return sub_DC7ED0(a1, v25, v47, 0, 0);
      }
LABEL_16:
      if ( *(_BYTE *)v6 != 17 )
        return (_QWORD *)v52;
      v12 = *(_DWORD *)(v6 + 32);
      v13 = v6 + 24;
LABEL_18:
      v26 = v12 <= 0x40 ? *(_QWORD *)(v6 + 24) == 0 : (unsigned int)sub_C444A0(v13) == v12;
      if ( !v26 || *v50 != 17 || !sub_9867B0((__int64)(v50 + 24)) )
        return (_QWORD *)v52;
      for ( i = sub_DD8400((__int64)a1, v5); *((_WORD *)i + 12) == 3; i = (__int64 *)i[4] )
        ;
      v28 = sub_D95540((__int64)i);
      v29 = sub_D97050((__int64)a1, v28);
      if ( v29 > sub_D97050((__int64)a1, a2) )
        return (_QWORD *)v52;
      v30 = sub_DD8400((__int64)a1, v49);
      if ( !(unsigned __int8)sub_D97230((__int64)v30, (__int64)i, 13) )
        return (_QWORD *)v52;
      v31 = sub_DC2CB0((__int64)a1, (__int64)i, a2);
      return sub_DCEE80(a1, (__int64)v31, (__int64)v30, 1);
    case '"':
    case '#':
    case '&':
    case '\'':
      goto LABEL_3;
    case '$':
    case '%':
    case '(':
    case ')':
      v5 = *(_QWORD *)(a3 - 32);
      v6 = *(_QWORD *)(a3 - 64);
LABEL_3:
      v9 = sub_D97050((__int64)a1, *(_QWORD *)(v5 + 8));
      if ( v9 > sub_D97050((__int64)a1, a2) )
        return (_QWORD *)v52;
      v48 = sub_B532B0(*(_WORD *)(a3 + 2) & 0x3F);
      v33 = sub_DD8400((__int64)a1, (__int64)v50);
      v51 = sub_DD8400((__int64)a1, v49);
      v34 = (__int64)sub_DD8400((__int64)a1, v5);
      v35 = (__int64)sub_DD8400((__int64)a1, v6);
      if ( *(_BYTE *)(sub_D95540((__int64)v33) + 8) != 14 )
        goto LABEL_37;
      if ( (__int64 *)v35 == v51 && (__int64 *)v34 == v33 )
      {
        if ( v48 )
          return (_QWORD *)sub_DCDFA0(a1, v34, v35, v36);
        else
          return (_QWORD *)sub_DCE050(a1, v34, v35, v36);
      }
      if ( (__int64 *)v35 == v33 && (__int64 *)v34 == v51 )
      {
        if ( v48 )
          return (_QWORD *)sub_DCE160(a1, v34, v35, v36);
        else
          return sub_DCEE80(a1, v34, v35, 0);
      }
LABEL_37:
      if ( *(_BYTE *)(sub_D95540(v34) + 8) != 14 || (v34 = sub_DD3750((__int64)a1, v34), !sub_D96A50(v34)) )
      {
        if ( v48 )
        {
          v34 = (__int64)sub_DD2D10((__int64)a1, v34, a2);
          if ( *(_BYTE *)(sub_D95540(v35) + 8) == 14 )
          {
            v35 = sub_DD3750((__int64)a1, v35);
            if ( sub_D96A50(v35) )
              goto LABEL_41;
          }
LABEL_40:
          v35 = (__int64)sub_DD2D10((__int64)a1, v35, a2);
          goto LABEL_41;
        }
        v34 = (__int64)sub_DC2CB0((__int64)a1, v34, a2);
        if ( *(_BYTE *)(sub_D95540(v35) + 8) == 14 )
        {
          v35 = sub_DD3750((__int64)a1, v35);
          if ( sub_D96A50(v35) )
            goto LABEL_41;
        }
LABEL_50:
        v35 = (__int64)sub_DC2CB0((__int64)a1, v35, a2);
        goto LABEL_41;
      }
      if ( *(_BYTE *)(sub_D95540(v35) + 8) != 14 || (v35 = sub_DD3750((__int64)a1, v35), !sub_D96A50(v35)) )
      {
        if ( v48 )
          goto LABEL_40;
        goto LABEL_50;
      }
LABEL_41:
      if ( !sub_D96A50(v34) && !sub_D96A50(v35) )
      {
        v37 = sub_DCC810(a1, (__int64)v33, v34, 0, 0);
        if ( v37 == sub_DCC810(a1, (__int64)v51, v35, 0, 0) )
        {
          if ( v48 )
            v40 = sub_DCDFA0(a1, v34, v35, v38);
          else
            v40 = sub_DCE050(a1, v34, v35, v38);
        }
        else
        {
          v37 = sub_DCC810(a1, (__int64)v33, v35, 0, 0);
          if ( v37 != sub_DCC810(a1, (__int64)v51, v34, 0, 0) )
            return (_QWORD *)v52;
          if ( v48 )
            v40 = sub_DCE160(a1, v34, v35, v39);
          else
            v40 = (__int64)sub_DCEE80(a1, v34, v35, 0);
        }
        return sub_DC7ED0(a1, v40, (__int64)v37, 0, 0);
      }
      return (_QWORD *)v52;
    default:
      return (_QWORD *)v52;
  }
}
