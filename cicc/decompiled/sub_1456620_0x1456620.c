// Function: sub_1456620
// Address: 0x1456620
//
__int64 __fastcall sub_1456620(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int16 v4; // cx
  const char *v5; // r14
  __int64 *v6; // r13
  __int64 *v7; // r15
  __int64 v8; // rdi
  __int64 v9; // r12
  const char *v10; // rsi
  __int64 result; // rax
  __int64 v12; // r13
  const char *v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  const char *v18; // rsi
  unsigned int v19; // r13d
  __int64 v20; // rax
  int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int16 v24; // ax
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // [rsp+8h] [rbp-48h] BYREF
  __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a2;
  v4 = *(_WORD *)(a1 + 24);
  switch ( v4 )
  {
    case 0:
      return sub_15537D0(*(_QWORD *)(a1 + 32), a2, 0);
    case 1:
      v12 = *(_QWORD *)(a1 + 32);
      v13 = "(trunc ";
      goto LABEL_14;
    case 2:
      v12 = *(_QWORD *)(a1 + 32);
      v13 = "(zext ";
      goto LABEL_14;
    case 3:
      v12 = *(_QWORD *)(a1 + 32);
      v13 = "(sext ";
LABEL_14:
      v14 = sub_1263B40(v2, v13);
      v15 = sub_1456040(v12);
      sub_154E060(v15, v14, 0, 0);
      v16 = sub_1263B40(v14, " ");
      sub_1456620(v12, v16);
      v2 = sub_1263B40(v16, " to ");
      sub_154E060(*(_QWORD *)(a1 + 40), v2, 0, 0);
      v10 = ")";
      goto LABEL_11;
    case 4:
    case 5:
    case 8:
    case 9:
      v5 = " umax ";
      if ( v4 != 8 )
      {
        v5 = " smax ";
        if ( v4 != 9 )
        {
          v5 = " + ";
          if ( v4 != 4 )
          {
            v5 = " * ";
            if ( v4 != 5 )
              v5 = 0;
          }
        }
      }
      sub_1263B40(a2, "(");
      v6 = *(__int64 **)(a1 + 32);
      v7 = &v6[*(_QWORD *)(a1 + 40)];
      if ( v7 != v6 )
      {
        while ( 1 )
        {
          v8 = *v6++;
          sub_1456620(v8, a2);
          if ( v7 == v6 )
            break;
          sub_1263B40(a2, v5);
        }
      }
      sub_1263B40(a2, ")");
      result = (unsigned int)*(unsigned __int16 *)(a1 + 24) - 4;
      if ( (unsigned int)result <= 1 )
      {
        result = *(unsigned __int16 *)(a1 + 26);
        if ( (result & 2) != 0 )
        {
          sub_1263B40(a2, "<nuw>");
          result = *(unsigned __int16 *)(a1 + 26);
        }
        v10 = "<nsw>";
        if ( (result & 4) != 0 )
          goto LABEL_11;
      }
      return result;
    case 6:
      v9 = sub_1263B40(a2, "(");
      sub_1456620(*(_QWORD *)(a1 + 32), v9);
      v2 = sub_1263B40(v9, " /u ");
      sub_1456620(*(_QWORD *)(a1 + 40), v2);
      v10 = ")";
      goto LABEL_11;
    case 7:
      v19 = 1;
      v20 = sub_1263B40(a2, "{");
      sub_1456620(**(_QWORD **)(a1 + 32), v20);
      v21 = *(_QWORD *)(a1 + 40);
      if ( v21 != 1 )
      {
        do
        {
          v22 = sub_1263B40(a2, ",+,");
          v23 = v19++;
          sub_1456620(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v23), v22);
        }
        while ( v21 != v19 );
      }
      sub_1263B40(a2, "}<");
      v24 = *(_WORD *)(a1 + 26);
      if ( (v24 & 2) != 0 )
      {
        sub_1263B40(a2, "nuw><");
        v24 = *(_WORD *)(a1 + 26);
      }
      if ( (v24 & 4) != 0 )
      {
        sub_1263B40(a2, "nsw><");
        v24 = *(_WORD *)(a1 + 26);
      }
      if ( (v24 & 1) != 0 && (v24 & 6) == 0 )
        sub_1263B40(a2, "nw><");
      sub_15537D0(**(_QWORD **)(*(_QWORD *)(a1 + 48) + 32LL), a2, 0);
      v10 = ">";
      goto LABEL_11;
    case 10:
      v17 = a1 - 32;
      v18 = "sizeof(";
      if ( sub_1456340(a1 - 32, &v27) )
        goto LABEL_35;
      if ( (unsigned __int8)sub_14563F0(v17, &v27) )
      {
        v18 = "alignof(";
LABEL_35:
        v25 = sub_1263B40(v2, v18);
        sub_154E060(v27, v25, 0, 0);
        result = sub_1263B40(v25, ")");
      }
      else if ( (unsigned __int8)sub_1456530(v17, &v28, v29) )
      {
        v26 = sub_1263B40(v2, "offsetof(");
        sub_154E060(v28, v26, 0, 0);
        sub_1263B40(v26, ", ");
        sub_15537D0(v29[0], v2, 0);
        result = sub_1263B40(v2, ")");
      }
      else
      {
        result = sub_15537D0(*(_QWORD *)(a1 - 8), v2, 0);
      }
      break;
    case 11:
      v10 = "***COULDNOTCOMPUTE***";
LABEL_11:
      result = sub_1263B40(v2, v10);
      break;
  }
  return result;
}
