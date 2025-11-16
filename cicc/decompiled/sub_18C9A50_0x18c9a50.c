// Function: sub_18C9A50
// Address: 0x18c9a50
//
__int64 __fastcall sub_18C9A50(__int64 *a1, int a2)
{
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 *v5; // rax
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // r13
  __int64 *v10; // rax
  __int64 v11; // r12
  __int64 *v12; // rax
  _QWORD *v13; // rdi
  __int64 *v14; // r13
  __int64 *v15; // rax
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 *v19; // r13
  __int64 *v20; // rax
  __int64 v21; // r12
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 *v26; // r13
  __int64 *v27; // rax
  __int64 v28; // r12
  __int64 *v29; // rax
  __int64 v30; // rdi
  __int64 *v31; // r12
  __int64 *v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r13
  __int64 *v36; // rax
  __int64 v37; // r12
  __int64 *v38; // rax
  __int64 v39; // rdi
  __int64 *v40; // r13
  __int64 *v41; // rax
  __int64 v42; // r12
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v46; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v47; // [rsp+10h] [rbp-30h] BYREF
  __int64 *v48; // [rsp+18h] [rbp-28h]

  switch ( a2 )
  {
    case 0:
      result = a1[1];
      if ( !result )
      {
        v9 = *(__int64 **)*a1;
        v10 = (__int64 *)sub_1643330(v9);
        v46 = (__int64 *)sub_1646BA0(v10, 0);
        v11 = sub_1644EA0(v46, &v46, 1, 0);
        v47 = 0;
        v12 = (__int64 *)sub_1563AB0((__int64 *)&v47, v9, -1, 30);
        v13 = (_QWORD *)*a1;
        v47 = v12;
        result = sub_1632080((__int64)v13, (__int64)"objc_autoreleaseReturnValue", 27, v11, (__int64)v12);
        a1[1] = result;
      }
      break;
    case 1:
      result = a1[2];
      if ( !result )
      {
        v14 = *(__int64 **)*a1;
        v15 = (__int64 *)sub_1643330(v14);
        v46 = (__int64 *)sub_1646BA0(v15, 0);
        v47 = 0;
        v16 = sub_1563AB0((__int64 *)&v47, v14, -1, 30);
        v17 = (__int64 *)sub_1643270(v14);
        v18 = sub_1644EA0(v17, &v46, 1, 0);
        result = sub_1632080(*a1, (__int64)"objc_release", 12, v18, v16);
        a1[2] = result;
      }
      break;
    case 2:
      result = a1[3];
      if ( !result )
      {
        v19 = *(__int64 **)*a1;
        v20 = (__int64 *)sub_1643330(v19);
        v46 = (__int64 *)sub_1646BA0(v20, 0);
        v21 = sub_1644EA0(v46, &v46, 1, 0);
        v47 = 0;
        v22 = (__int64 *)sub_1563AB0((__int64 *)&v47, v19, -1, 30);
        v23 = *a1;
        v47 = v22;
        result = sub_1632080(v23, (__int64)"objc_retain", 11, v21, (__int64)v22);
        a1[3] = result;
      }
      break;
    case 3:
      result = a1[4];
      if ( !result )
      {
        v24 = (__int64 *)sub_1643330(*(_QWORD **)*a1);
        v47 = (__int64 *)sub_1646BA0(v24, 0);
        v25 = sub_1644EA0(v47, &v47, 1, 0);
        result = sub_1632080(*a1, (__int64)"objc_retainBlock", 16, v25, 0);
        a1[4] = result;
      }
      break;
    case 4:
      result = a1[5];
      if ( !result )
      {
        v26 = *(__int64 **)*a1;
        v27 = (__int64 *)sub_1643330(v26);
        v46 = (__int64 *)sub_1646BA0(v27, 0);
        v28 = sub_1644EA0(v46, &v46, 1, 0);
        v47 = 0;
        v29 = (__int64 *)sub_1563AB0((__int64 *)&v47, v26, -1, 30);
        v30 = *a1;
        v47 = v29;
        result = sub_1632080(v30, (__int64)"objc_autorelease", 16, v28, (__int64)v29);
        a1[5] = result;
      }
      break;
    case 5:
      result = a1[6];
      if ( !result )
      {
        v31 = *(__int64 **)*a1;
        v32 = (__int64 *)sub_1643330(v31);
        v48 = (__int64 *)sub_1646BA0(v32, 0);
        v47 = (__int64 *)sub_1646BA0(v48, 0);
        v46 = 0;
        v45 = sub_1563AB0((__int64 *)&v46, v31, -1, 30);
        v45 = sub_1563AB0(&v45, v31, 1, 22);
        v33 = (__int64 *)sub_1643270(v31);
        v34 = sub_1644EA0(v33, &v47, 2, 0);
        result = sub_1632080(*a1, (__int64)"objc_storeStrong", 16, v34, v45);
        a1[6] = result;
      }
      break;
    case 6:
      result = a1[7];
      if ( !result )
      {
        v35 = *(__int64 **)*a1;
        v36 = (__int64 *)sub_1643330(v35);
        v46 = (__int64 *)sub_1646BA0(v36, 0);
        v37 = sub_1644EA0(v46, &v46, 1, 0);
        v47 = 0;
        v38 = (__int64 *)sub_1563AB0((__int64 *)&v47, v35, -1, 30);
        v39 = *a1;
        v47 = v38;
        result = sub_1632080(v39, (__int64)"objc_retainAutoreleasedReturnValue", 34, v37, (__int64)v38);
        a1[7] = result;
      }
      break;
    case 7:
      result = a1[8];
      if ( !result )
      {
        v40 = *(__int64 **)*a1;
        v41 = (__int64 *)sub_1643330(v40);
        v46 = (__int64 *)sub_1646BA0(v41, 0);
        v42 = sub_1644EA0(v46, &v46, 1, 0);
        v47 = 0;
        v43 = (__int64 *)sub_1563AB0((__int64 *)&v47, v40, -1, 30);
        v44 = *a1;
        v47 = v43;
        result = sub_1632080(v44, (__int64)"objc_retainAutorelease", 22, v42, (__int64)v43);
        a1[8] = result;
      }
      break;
    case 8:
      result = a1[9];
      if ( !result )
      {
        v4 = *(__int64 **)*a1;
        v5 = (__int64 *)sub_1643330(v4);
        v46 = (__int64 *)sub_1646BA0(v5, 0);
        v6 = sub_1644EA0(v46, &v46, 1, 0);
        v47 = 0;
        v7 = (__int64 *)sub_1563AB0((__int64 *)&v47, v4, -1, 30);
        v8 = *a1;
        v47 = v7;
        result = sub_1632080(v8, (__int64)"objc_retainAutoreleaseReturnValue", 33, v6, (__int64)v7);
        a1[9] = result;
      }
      break;
  }
  return result;
}
