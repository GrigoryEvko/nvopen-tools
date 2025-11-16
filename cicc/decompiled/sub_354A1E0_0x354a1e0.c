// Function: sub_354A1E0
// Address: 0x354a1e0
//
_QWORD *__fastcall sub_354A1E0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 *a4)
{
  __int64 v8; // rcx
  _QWORD *v9; // r15
  _QWORD *v10; // r8
  _QWORD *v11; // rax
  _QWORD *v12; // r10
  _QWORD *v13; // rdx
  _QWORD *v14; // r11
  __int64 v15; // rsi
  __int64 v16; // rdi
  _QWORD *v17; // rsi
  _QWORD **v18; // rdi
  __int64 v19; // rdi
  __int64 v21; // rax
  _QWORD *v22; // r8
  _QWORD *v23; // r10
  __int64 v24; // r9
  __int64 v25; // [rsp+0h] [rbp-90h]
  __int64 v26; // [rsp+0h] [rbp-90h]
  _QWORD *v27; // [rsp+8h] [rbp-88h]
  _QWORD *v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  _QWORD *v30; // [rsp+10h] [rbp-80h]
  _QWORD *v31; // [rsp+10h] [rbp-80h]
  _QWORD *v32; // [rsp+10h] [rbp-80h]
  _QWORD *v33; // [rsp+18h] [rbp-78h]
  _QWORD *v34; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v35; // [rsp+28h] [rbp-68h]
  _QWORD *v36; // [rsp+30h] [rbp-60h]
  __int64 v37; // [rsp+38h] [rbp-58h]
  _QWORD v38[3]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v39; // [rsp+58h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 24);
  v9 = (_QWORD *)*a3;
  v10 = (_QWORD *)a3[1];
  v11 = *(_QWORD **)a2;
  v12 = (_QWORD *)a3[2];
  v39 = a3[3];
  v13 = *(_QWORD **)(a2 + 16);
  v14 = *(_QWORD **)(a2 + 8);
  v38[0] = v9;
  v38[1] = v10;
  v38[2] = v12;
  v34 = v11;
  v35 = v14;
  v36 = v13;
  v37 = v8;
  v15 = (v9 - v10 + ((((v39 - v8) >> 3) - 1) << 6) + v13 - v11) >> 2;
  if ( v15 <= 0 )
  {
LABEL_27:
    v30 = v10;
    v25 = v39;
    v27 = v12;
    v21 = sub_3549CC0(v38, &v34);
    v22 = v30;
    v23 = v27;
    v24 = v25;
    if ( v21 != 2 )
    {
      if ( v21 != 3 )
      {
        if ( v21 != 1 )
        {
LABEL_30:
          v8 = v24;
          v13 = v23;
          v14 = v22;
          v11 = v9;
          goto LABEL_16;
        }
LABEL_36:
        v11 = v34;
        if ( *v34 != *a4 )
        {
          v29 = v24;
          v32 = v23;
          v33 = v22;
          sub_3546A50((__int64 *)&v34);
          v24 = v29;
          v23 = v32;
          v22 = v33;
          goto LABEL_30;
        }
LABEL_38:
        v14 = v35;
        v13 = v36;
        v8 = v37;
        goto LABEL_16;
      }
      v11 = v34;
      if ( *v34 == *a4 )
        goto LABEL_38;
      sub_3546A50((__int64 *)&v34);
      v24 = v25;
      v23 = v27;
      v22 = v30;
    }
    v11 = v34;
    if ( *v34 == *a4 )
      goto LABEL_38;
    v26 = v24;
    v28 = v23;
    v31 = v22;
    sub_3546A50((__int64 *)&v34);
    v24 = v26;
    v23 = v28;
    v22 = v31;
    goto LABEL_36;
  }
  v16 = *a4;
  while ( *v11 != v16 )
  {
    v34 = ++v11;
    if ( v11 == v13 )
    {
      v37 = v8 + 8;
      v11 = *(_QWORD **)(v8 + 8);
      v8 += 8;
      v13 = v11 + 64;
      v35 = v11;
      v14 = v11;
      v36 = v11 + 64;
      v34 = v11;
      if ( v16 == *v11 )
        break;
    }
    else if ( v16 == *v11 )
    {
      break;
    }
    v34 = ++v11;
    if ( v11 == v13 )
    {
      v37 = v8 + 8;
      v11 = *(_QWORD **)(v8 + 8);
      v8 += 8;
      v13 = v11 + 64;
      v35 = v11;
      v14 = v11;
      v36 = v11 + 64;
      v34 = v11;
    }
    if ( v16 == *v11 )
      break;
    v34 = ++v11;
    if ( v13 == v11 )
    {
      v37 = v8 + 8;
      v11 = *(_QWORD **)(v8 + 8);
      v8 += 8;
      v13 = v11 + 64;
      v35 = v11;
      v14 = v11;
      v36 = v11 + 64;
      v34 = v11;
    }
    if ( v16 == *v11 )
      break;
    v34 = v11 + 1;
    if ( v13 == v11 + 1 )
    {
      v37 = v8 + 8;
      v35 = *(_QWORD **)(v8 + 8);
      v36 = v35 + 64;
      v34 = v35;
      if ( !--v15 )
        goto LABEL_27;
    }
    else if ( !--v15 )
    {
      goto LABEL_27;
    }
    v11 = v34;
    v14 = v35;
    v13 = v36;
    v8 = v37;
  }
LABEL_16:
  *(_QWORD *)a2 = v11;
  *(_QWORD *)(a2 + 8) = v14;
  *(_QWORD *)(a2 + 16) = v13;
  *(_QWORD *)(a2 + 24) = v8;
  if ( (_QWORD *)*a3 != v11 )
  {
    v17 = v11 + 1;
    *(_QWORD *)a2 = v11 + 1;
    if ( v13 == v11 + 1 )
    {
      *(_QWORD *)(a2 + 24) = v8 + 8;
      v17 = *(_QWORD **)(v8 + 8);
      *(_QWORD *)(a2 + 8) = v17;
      *(_QWORD *)(a2 + 16) = v17 + 64;
      *(_QWORD *)a2 = v17;
    }
    while ( (_QWORD *)*a3 != v17 )
    {
      while ( 1 )
      {
        if ( *v17 != *a4 )
        {
          *v11++ = *v17;
          if ( v11 == v13 )
          {
            v11 = *(_QWORD **)(v8 + 8);
            v17 = *(_QWORD **)a2;
            v8 += 8;
            v13 = v11 + 64;
            v14 = v11;
          }
          else
          {
            v17 = *(_QWORD **)a2;
          }
        }
        *(_QWORD *)a2 = ++v17;
        if ( v17 != *(_QWORD **)(a2 + 16) )
          break;
        v18 = (_QWORD **)(*(_QWORD *)(a2 + 24) + 8LL);
        *(_QWORD *)(a2 + 24) = v18;
        v17 = *v18;
        v19 = (__int64)(*v18 + 64);
        *(_QWORD *)(a2 + 8) = v17;
        *(_QWORD *)(a2 + 16) = v19;
        *(_QWORD *)a2 = v17;
        if ( (_QWORD *)*a3 == v17 )
          goto LABEL_25;
      }
    }
  }
LABEL_25:
  *a1 = v11;
  a1[1] = v14;
  a1[2] = v13;
  a1[3] = v8;
  return a1;
}
