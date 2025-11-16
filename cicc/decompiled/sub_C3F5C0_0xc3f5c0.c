// Function: sub_C3F5C0
// Address: 0xc3f5c0
//
__int64 __fastcall sub_C3F5C0(__int64 a1, __int64 *a2, unsigned int a3)
{
  unsigned int v6; // r13d
  _QWORD *v8; // r14
  __int64 *v9; // rsi
  __int64 *v10; // rsi
  __int64 *v11; // rsi
  __int64 *v12; // rsi
  char v13; // al
  __int64 v14; // rdi
  int v15; // eax
  int v16; // r13d
  int v17; // eax
  int v18; // r13d
  int v19; // eax
  int v20; // r13d
  int v21; // eax
  int v22; // r13d
  int v23; // eax
  int v24; // r13d
  int v25; // eax
  __int64 *v26; // rdi
  char v27; // al
  int v28; // eax
  int v29; // r13d
  int v30; // eax
  void **v31; // rdi
  _QWORD *v32; // rdi
  _QWORD *v33; // [rsp+18h] [rbp-158h]
  _QWORD *v34; // [rsp+18h] [rbp-158h]
  _QWORD *v35[4]; // [rsp+40h] [rbp-130h] BYREF
  _QWORD *v36[4]; // [rsp+60h] [rbp-110h] BYREF
  _QWORD v37[4]; // [rsp+80h] [rbp-F0h] BYREF
  _QWORD v38[4]; // [rsp+A0h] [rbp-D0h] BYREF
  _QWORD *v39[2]; // [rsp+C0h] [rbp-B0h] BYREF
  char v40; // [rsp+D4h] [rbp-9Ch]
  __int64 v41[4]; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v42[4]; // [rsp+100h] [rbp-70h] BYREF
  _QWORD *v43[2]; // [rsp+120h] [rbp-50h] BYREF
  char v44; // [rsp+134h] [rbp-3Ch]

  if ( (unsigned int)sub_C3CE50(a1) == 1 )
    goto LABEL_7;
  if ( (unsigned int)sub_C3CE50((__int64)a2) == 1 )
    goto LABEL_14;
  if ( (unsigned int)sub_C3CE50(a1) == 3 )
  {
    if ( !(unsigned int)sub_C3CE50((__int64)a2) )
      goto LABEL_10;
    if ( (unsigned int)sub_C3CE50(a1) )
      goto LABEL_5;
LABEL_9:
    if ( (unsigned int)sub_C3CE50((__int64)a2) != 3 )
      goto LABEL_5;
LABEL_10:
    sub_C3D480(a1, 0, 0, 0);
    return 0;
  }
  if ( !(unsigned int)sub_C3CE50(a1) )
    goto LABEL_9;
LABEL_5:
  if ( (unsigned int)sub_C3CE50(a1) != 3 && (unsigned int)sub_C3CE50(a1) )
  {
    if ( (unsigned int)sub_C3CE50((__int64)a2) != 3 && (unsigned int)sub_C3CE50((__int64)a2) )
    {
      v8 = sub_C33340();
      v9 = *(__int64 **)(a1 + 8);
      if ( (_QWORD *)*v9 == v8 )
        sub_C3C790(v35, (_QWORD **)v9);
      else
        sub_C33EB0(v35, v9);
      v10 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24LL);
      if ( (_QWORD *)*v10 == v8 )
        sub_C3C790(v36, (_QWORD **)v10);
      else
        sub_C33EB0(v36, v10);
      v11 = (__int64 *)a2[1];
      if ( (_QWORD *)*v11 == v8 )
        sub_C3C790(v37, (_QWORD **)v11);
      else
        sub_C33EB0(v37, v11);
      v12 = (__int64 *)(a2[1] + 24);
      if ( (_QWORD *)*v12 == v8 )
        sub_C3C790(v38, (_QWORD **)v12);
      else
        sub_C33EB0(v38, v12);
      if ( v35[0] == v8 )
        sub_C3C790(v39, v35);
      else
        sub_C33EB0(v39, (__int64 *)v35);
      if ( v39[0] == v8 )
        v6 = sub_C3F5C0(v39, v37, a3);
      else
        v6 = sub_C3B950((__int64)v39, (__int64)v37, a3);
      if ( v8 == v39[0] )
      {
        v13 = *((_BYTE *)v39[1] + 20) & 7;
        if ( v13 == 1 )
        {
LABEL_31:
          sub_C3CBE0(*(__int64 **)(a1 + 8), (__int64 *)v39);
          v14 = *(_QWORD *)(a1 + 8);
          if ( v8 == *(_QWORD **)(v14 + 24) )
            sub_C3CEB0((void **)(v14 + 24), 0);
          else
            sub_C37310(v14 + 24, 0);
          goto LABEL_33;
        }
      }
      else
      {
        v13 = v40 & 7;
        if ( (v40 & 7) == 1 )
          goto LABEL_31;
      }
      if ( v13 == 3 || !v13 )
        goto LABEL_31;
      if ( v8 == v35[0] )
        sub_C3C790(v41, v35);
      else
        sub_C33EB0(v41, (__int64 *)v35);
      if ( v8 == v39[0] )
        sub_C3CCB0((__int64)v39);
      else
        sub_C34440((unsigned __int8 *)v39);
      if ( v8 == (_QWORD *)v41[0] )
        v15 = sub_C3F220(v41, (__int64)v37, (__int64)v39, a3);
      else
        v15 = sub_C3B3E0((__int64)v41, (__int64)v37, (__int64)v39, a3);
      v16 = v15 | v6;
      if ( v8 == v39[0] )
        sub_C3CCB0((__int64)v39);
      else
        sub_C34440((unsigned __int8 *)v39);
      if ( v8 == v35[0] )
        sub_C3C790(v42, v35);
      else
        sub_C33EB0(v42, (__int64 *)v35);
      if ( v8 == (_QWORD *)v42[0] )
        v17 = sub_C3F5C0(v42, v38, a3);
      else
        v17 = sub_C3B950((__int64)v42, (__int64)v38, a3);
      v18 = v17 | v16;
      if ( v8 == v36[0] )
        sub_C3C790(v43, v36);
      else
        sub_C33EB0(v43, (__int64 *)v36);
      if ( v8 == v43[0] )
        v19 = sub_C3F5C0(v43, v37, a3);
      else
        v19 = sub_C3B950((__int64)v43, (__int64)v37, a3);
      v20 = v19 | v18;
      if ( v8 == (_QWORD *)v42[0] )
        v21 = sub_C3D800(v42, (__int64)v43, a3);
      else
        v21 = sub_C3ADF0((__int64)v42, (__int64)v43, a3);
      v22 = v21 | v20;
      if ( v8 == (_QWORD *)v41[0] )
        v23 = sub_C3D800(v41, (__int64)v42, a3);
      else
        v23 = sub_C3ADF0((__int64)v41, (__int64)v42, a3);
      v24 = v23 | v22;
      sub_91D830(v43);
      sub_91D830(v42);
      if ( v8 == v39[0] )
        sub_C3C790(v43, v39);
      else
        sub_C33EB0(v43, (__int64 *)v39);
      if ( v8 == v43[0] )
        v25 = sub_C3D800((__int64 *)v43, (__int64)v41, a3);
      else
        v25 = sub_C3ADF0((__int64)v43, (__int64)v41, a3);
      v26 = *(__int64 **)(a1 + 8);
      v6 = v25 | v24;
      if ( v8 == (_QWORD *)*v26 )
      {
        if ( v43[0] == v8 )
        {
          sub_C3C9E0(v26, (__int64 *)v43);
          goto LABEL_64;
        }
        if ( v26 == (__int64 *)v43 )
        {
LABEL_65:
          v27 = v44 & 7;
          if ( (v44 & 7) == 1 )
            goto LABEL_74;
LABEL_66:
          if ( v27 )
          {
            if ( v8 == v39[0] )
              v28 = sub_C3D820((__int64 *)v39, (__int64)v43, a3);
            else
              v28 = sub_C3B1F0((__int64)v39, (__int64)v43, a3);
            v29 = v28 | v6;
            if ( v8 == v39[0] )
              v30 = sub_C3D800((__int64 *)v39, (__int64)v41, a3);
            else
              v30 = sub_C3ADF0((__int64)v39, (__int64)v41, a3);
            v6 = v30 | v29;
            sub_C3CBE0((__int64 *)(*(_QWORD *)(a1 + 8) + 24LL), (__int64 *)v39);
            goto LABEL_72;
          }
LABEL_74:
          v31 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
          if ( v8 == *v31 )
            sub_C3CEB0(v31, 0);
          else
            sub_C37310((__int64)v31, 0);
LABEL_72:
          sub_91D830(v43);
          sub_91D830(v41);
LABEL_33:
          sub_91D830(v39);
          sub_91D830(v38);
          sub_91D830(v37);
          sub_91D830(v36);
          sub_91D830(v35);
          return v6;
        }
        v34 = *(_QWORD **)(a1 + 8);
        sub_969EE0((__int64)v26);
        v32 = v34;
      }
      else
      {
        if ( v43[0] != v8 )
        {
          sub_C33E70(v26, (__int64 *)v43);
          goto LABEL_64;
        }
        if ( v26 == (__int64 *)v43 )
          goto LABEL_73;
        v33 = *(_QWORD **)(a1 + 8);
        sub_C338F0((__int64)v26);
        v32 = v33;
      }
      if ( v8 == v43[0] )
        sub_C3C790(v32, v43);
      else
        sub_C33EB0(v32, (__int64 *)v43);
LABEL_64:
      if ( v8 != v43[0] )
        goto LABEL_65;
LABEL_73:
      v27 = *((_BYTE *)v43[1] + 20) & 7;
      if ( v27 == 1 )
        goto LABEL_74;
      goto LABEL_66;
    }
LABEL_14:
    v6 = 0;
    sub_C3C9E0((__int64 *)a1, a2);
    return v6;
  }
LABEL_7:
  v6 = 0;
  sub_C3C9E0((__int64 *)a1, (__int64 *)a1);
  return v6;
}
