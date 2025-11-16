// Function: sub_16A1610
// Address: 0x16a1610
//
__int64 __fastcall sub_16A1610(__int64 *a1, __int64 *a2, unsigned int a3, double a4, double a5, double a6)
{
  int v8; // eax
  int v9; // r14d
  char v10; // dl
  int v11; // eax
  int v12; // eax
  int v13; // r14d
  int v14; // eax
  int v15; // ebx
  int v16; // ebx
  int v17; // ebx
  int v18; // ebx
  char v19; // al
  char *v20; // rax
  char v21; // al
  char *v22; // rax
  char v23; // al
  char *v24; // rax
  char v25; // al
  char *v26; // rax
  char v27; // al
  int v28; // [rsp+4h] [rbp-18Ch]
  __int16 *v29; // [rsp+40h] [rbp-150h]
  __int16 *v30; // [rsp+58h] [rbp-138h]
  int v31; // [rsp+58h] [rbp-138h]
  unsigned int v32; // [rsp+58h] [rbp-138h]
  int v33; // [rsp+58h] [rbp-138h]
  __int64 v34[4]; // [rsp+68h] [rbp-128h] BYREF
  __int64 v35[3]; // [rsp+88h] [rbp-108h] BYREF
  char v36[8]; // [rsp+A0h] [rbp-F0h] BYREF
  _QWORD v37[3]; // [rsp+A8h] [rbp-E8h] BYREF
  char v38[8]; // [rsp+C0h] [rbp-D0h] BYREF
  _QWORD v39[3]; // [rsp+C8h] [rbp-C8h] BYREF
  char v40[8]; // [rsp+E0h] [rbp-B0h] BYREF
  __int16 *v41[2]; // [rsp+E8h] [rbp-A8h] BYREF
  char v42; // [rsp+FAh] [rbp-96h]
  char v43[8]; // [rsp+100h] [rbp-90h] BYREF
  __int16 *v44[3]; // [rsp+108h] [rbp-88h] BYREF
  char v45[8]; // [rsp+120h] [rbp-70h] BYREF
  __int16 *v46[3]; // [rsp+128h] [rbp-68h] BYREF
  char v47[8]; // [rsp+140h] [rbp-50h] BYREF
  __int16 *v48[2]; // [rsp+148h] [rbp-48h] BYREF
  char v49; // [rsp+15Ah] [rbp-36h]

  if ( (unsigned int)sub_169C920((__int64)a1) == 3 && !(unsigned int)sub_169C920((__int64)a2)
    || !(unsigned int)sub_169C920((__int64)a1) && (unsigned int)sub_169C920((__int64)a2) == 3 )
  {
    sub_169CAA0((__int64)a1, 0, 0, 0, *(float *)&a4);
    return 0;
  }
  if ( (unsigned int)sub_169C920((__int64)a1) == 3 || !(unsigned int)sub_169C920((__int64)a1) )
  {
    sub_16A0170(a1, a1);
    return 0;
  }
  if ( (unsigned int)sub_169C920((__int64)a2) == 3 || !(unsigned int)sub_169C920((__int64)a2) )
  {
    sub_16A0170(a1, a2);
    return 0;
  }
  sub_169C7A0(v34, (__int64 *)(a1[1] + 8));
  sub_169C7A0(v35, (__int64 *)(a1[1] + 40));
  sub_169C7A0(v37, (__int64 *)(a2[1] + 8));
  sub_169C7A0(v39, (__int64 *)(a2[1] + 40));
  sub_169C7A0(v41, v34);
  v30 = (__int16 *)sub_16982C0();
  if ( v41[0] == v30 )
  {
    v8 = sub_16A1EA0(v41, v37, a3);
    v9 = v8;
  }
  else if ( (unsigned __int8)sub_169DE70((__int64)v40) || (unsigned __int8)sub_169DE70((__int64)v36) )
  {
    if ( v30 == v41[0] )
      sub_169CAA0((__int64)v41, 0, 0, 0, *(float *)&a4);
    else
      sub_16986F0(v41, 0, 0, 0);
    v9 = 1;
    v8 = 1;
  }
  else if ( v41[0] == sub_1698270()
         && ((v22 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v23 = qword_4FBB490[2]) : (v23 = *v22), v23) )
  {
    v8 = sub_169F220((__int64)v40, (__int64)v36, a3);
    v9 = v8;
  }
  else
  {
    v8 = sub_169DCA0(v41, v37, a3);
    v9 = v8;
  }
  if ( v30 == v41[0] )
  {
    v10 = v41[1][13] & 7;
    if ( v10 == 1 )
      goto LABEL_20;
LABEL_26:
    if ( v10 == 3 || !v10 )
      goto LABEL_20;
    sub_169C7A0(v44, v34);
    if ( v30 == v41[0] )
      sub_169C8D0((__int64)v41, a4, a5, a6);
    else
      sub_1699490((__int64)v41);
    v29 = v44[0];
    if ( v30 == v44[0] )
    {
      v12 = sub_169F930(v44, (__int64)v37, (__int64)v41, a3);
    }
    else if ( v29 == sub_1698270()
           && ((v20 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v21 = qword_4FBB490[2]) : (v21 = *v20), v21) )
    {
      v12 = sub_169F510((__int64)v43, (__int64)v36, (__int64)v40, a3);
    }
    else
    {
      v12 = sub_169DD30(v44, v37, v41, a3);
    }
    v13 = v12 | v9;
    if ( v30 == v41[0] )
      sub_169C8D0((__int64)v41, a4, a5, a6);
    else
      sub_1699490((__int64)v41);
    sub_169C7A0(v46, v34);
    if ( v30 == v46[0] )
    {
      v14 = sub_16A1EA0(v46, v39, a3);
    }
    else if ( (unsigned __int8)sub_169DE70((__int64)v45) || (unsigned __int8)sub_169DE70((__int64)v38) )
    {
      if ( v30 == v46[0] )
        sub_169CAA0((__int64)v46, 0, 0, 0, *(float *)&a4);
      else
        sub_16986F0(v46, 0, 0, 0);
      v14 = 1;
    }
    else if ( v46[0] == sub_1698270()
           && ((v26 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v27 = qword_4FBB490[2]) : (v27 = *v26), v27) )
    {
      v14 = sub_169F220((__int64)v45, (__int64)v38, a3);
    }
    else
    {
      v14 = sub_169DCA0(v46, v39, a3);
    }
    v28 = v14 | v13;
    sub_169C7A0(v48, v35);
    if ( v30 == v48[0] )
    {
      v15 = sub_16A1EA0(v48, v37, a3);
    }
    else if ( (unsigned __int8)sub_169DE70((__int64)v47) || (unsigned __int8)sub_169DE70((__int64)v36) )
    {
      if ( v30 == v48[0] )
        sub_169CAA0((__int64)v48, 0, 0, 0, *(float *)&a4);
      else
        sub_16986F0(v48, 0, 0, 0);
      v15 = 1;
    }
    else if ( v48[0] == sub_1698270()
           && ((v24 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v25 = qword_4FBB490[2]) : (v25 = *v24), v25) )
    {
      v15 = sub_169F220((__int64)v47, (__int64)v36, a3);
    }
    else
    {
      v15 = sub_169DCA0(v48, v37, a3);
    }
    v16 = sub_16A0E20((__int64)v45, (__int64)v47, a3, a4, a5, a6) | v28 | v15;
    v17 = sub_16A0E20((__int64)v43, (__int64)v45, a3, a4, a5, a6) | v16;
    sub_127D120(v48);
    sub_127D120(v46);
    sub_169C7A0(v48, (__int64 *)v41);
    v18 = sub_16A0E20((__int64)v47, (__int64)v43, a3, a4, a5, a6) | v17;
    sub_16A0360((__int64 *)(a1[1] + 8), (__int64 *)v48);
    if ( v30 == v48[0] )
    {
      v19 = v48[1][13] & 7;
      if ( v19 == 1 )
        goto LABEL_51;
    }
    else
    {
      v19 = v49 & 7;
      if ( (v49 & 7) == 1 )
        goto LABEL_51;
    }
    if ( v19 )
    {
      v33 = sub_16A1420((__int64)v40, (__int64)v47, a3, a4, a5, a6);
      v18 |= v33 | sub_16A0E20((__int64)v40, (__int64)v43, a3, a4, a5, a6);
      sub_16A0360((__int64 *)(a1[1] + 40), (__int64 *)v41);
LABEL_49:
      sub_127D120(v48);
      sub_127D120(v44);
      v11 = v18;
      goto LABEL_21;
    }
LABEL_51:
    sub_169C9F0(a1[1] + 32, 0);
    goto LABEL_49;
  }
  v10 = v42 & 7;
  if ( (v42 & 7) != 1 )
    goto LABEL_26;
LABEL_20:
  v31 = v8;
  sub_16A0360((__int64 *)(a1[1] + 8), (__int64 *)v41);
  sub_169C9F0(a1[1] + 32, 0);
  v11 = v31;
LABEL_21:
  v32 = v11;
  sub_127D120(v41);
  sub_127D120(v39);
  sub_127D120(v37);
  sub_127D120(v35);
  sub_127D120(v34);
  return v32;
}
