// Function: sub_128F9F0
// Address: 0x128f9f0
//
_QWORD *__fastcall sub_128F9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rsi
  __int64 v7; // r14
  char *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  char *v12; // rdx
  __int64 *v14; // rbx
  char v15; // al
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 *v19; // r13
  __int64 v20; // rdi
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned int v25; // r14d
  int v26; // eax
  __int64 v27; // rdi
  char *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdi
  char *v31; // [rsp+8h] [rbp-78h]
  char *v32; // [rsp+8h] [rbp-78h]
  char *v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+18h] [rbp-68h] BYREF
  char *v35; // [rsp+20h] [rbp-60h] BYREF
  char v36; // [rsp+30h] [rbp-50h]
  char v37; // [rsp+31h] [rbp-4Fh]
  _BYTE v38[16]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v39; // [rsp+50h] [rbp-30h]

  v6 = *(_QWORD *)(a2 + 72);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = sub_128D0F0((__int64 **)a1, v6, a3, a4, a5);
  v12 = sub_128D0F0((__int64 **)a1, v7, v9, v10, v11);
  switch ( *(_BYTE *)(a2 + 56) )
  {
    case '\'':
      return (_QWORD *)sub_1288F60(a1, v8, (__int64)v12, *(_QWORD *)a2);
    case '(':
      return (_QWORD *)sub_1288370(a1, v8, (__int64)v12, *(_QWORD *)a2);
    case ')':
      return (_QWORD *)sub_1288770(a1, v8, (__int64)v12, *(_QWORD *)a2);
    case '*':
      return (_QWORD *)sub_1289D20((__int64 *)a1, (__int64)v8, (__int64)v12, *(_QWORD *)a2);
    case '+':
      return sub_1288DC0(a1, (__int64)v8, (__int64)v12, *(_QWORD *)a2);
    case '5':
      return (_QWORD *)sub_1288B70(a1, (__int64 *)v8, (__int64)v12);
    case '6':
      return (_QWORD *)sub_1289360(a1, (__int64 *)v8, (__int64)v12, *(_QWORD *)a2);
    case '7':
      v37 = 1;
      v14 = *(__int64 **)(a1 + 8);
      v35 = "and";
      v36 = 3;
      v15 = v12[16];
      if ( (unsigned __int8)v15 > 0x10u )
        goto LABEL_33;
      if ( v15 != 13 )
        goto LABEL_13;
      v25 = *((_DWORD *)v12 + 8);
      if ( v25 <= 0x40 )
      {
        v16 = v8;
        if ( *((_QWORD *)v12 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25) )
          return v16;
      }
      else
      {
        v33 = v12;
        v16 = v8;
        v26 = sub_16A58F0(v12 + 24);
        v12 = v33;
        if ( v25 == v26 )
          return v16;
      }
LABEL_13:
      if ( (unsigned __int8)v8[16] <= 0x10u )
        return (_QWORD *)sub_15A2CF0(v8, v12);
LABEL_33:
      v27 = 26;
      v39 = 257;
      v28 = v8;
      goto LABEL_34;
    case '8':
      v37 = 1;
      v14 = *(__int64 **)(a1 + 8);
      v35 = "or";
      v36 = 3;
      if ( (unsigned __int8)v12[16] > 0x10u )
        goto LABEL_36;
      v31 = v12;
      v16 = v8;
      if ( (unsigned __int8)sub_1593BB0(v12) )
        return v16;
      v12 = v31;
      if ( (unsigned __int8)v8[16] <= 0x10u )
        return (_QWORD *)sub_15A2D10(v8, v31);
LABEL_36:
      v28 = v8;
      v27 = 27;
      v39 = 257;
LABEL_34:
      v29 = sub_15FB440(v27, v28, v12, v38, 0);
      v30 = v14[1];
      v16 = (_QWORD *)v29;
      if ( v30 )
      {
        v19 = (unsigned __int64 *)v14[2];
        v20 = v30 + 40;
LABEL_24:
        sub_157E9D0(v20, v16);
        v21 = *v19;
        v22 = v16[3];
        v16[4] = v19;
        v21 &= 0xFFFFFFFFFFFFFFF8LL;
        v16[3] = v21 | v22 & 7;
        *(_QWORD *)(v21 + 8) = v16 + 3;
        *v19 = *v19 & 7 | (unsigned __int64)(v16 + 3);
      }
LABEL_25:
      sub_164B780(v16, &v35);
      v23 = *v14;
      if ( *v14 )
      {
        v34 = *v14;
        sub_1623A60(&v34, v23, 2);
        if ( v16[6] )
          sub_161E7C0(v16 + 6);
        v24 = v34;
        v16[6] = v34;
        if ( v24 )
          sub_1623210(&v34, v24, v16 + 6);
      }
      return v16;
    case '9':
      v37 = 1;
      v14 = *(__int64 **)(a1 + 8);
      v35 = "xor";
      v36 = 3;
      if ( (unsigned __int8)v8[16] <= 0x10u && (unsigned __int8)v12[16] <= 0x10u )
      {
        v32 = v12;
        v17 = sub_15A2A30(28, v8, v12, 0, 0);
        v12 = v32;
        v16 = (_QWORD *)v17;
        if ( v17 )
          return v16;
      }
      v39 = 257;
      v16 = (_QWORD *)sub_15FB440(28, v8, v12, v38, 0);
      v18 = v14[1];
      if ( !v18 )
        goto LABEL_25;
      v19 = (unsigned __int64 *)v14[2];
      v20 = v18 + 40;
      goto LABEL_24;
    default:
      sub_127B550("unsupported binary expression!", (_DWORD *)(a2 + 36), 1);
  }
}
