// Function: sub_33809B0
// Address: 0x33809b0
//
_QWORD *__fastcall sub_33809B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r11
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rsi
  int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // r13
  int v21; // r12d
  _QWORD *result; // rax
  __int64 v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 *v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+40h] [rbp-40h] BYREF
  int v29; // [rsp+48h] [rbp-38h]

  v8 = a1[108];
  v9 = *(_QWORD *)(v8 + 16);
  v25 = *(__int64 **)(a2 + 8);
  v10 = sub_2E79000(*(__int64 **)(v8 + 40));
  v11 = sub_2D5BAE0(v9, v10, v25, 1);
  v28 = 0;
  v13 = a1[108];
  v14 = v12;
  v15 = v11;
  v16 = *a1;
  v29 = *((_DWORD *)a1 + 212);
  if ( v16 )
  {
    if ( &v28 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v28 = v17;
      if ( v17 )
      {
        v23 = v15;
        v24 = v12;
        v26 = v13;
        sub_B96E90((__int64)&v28, v17, 1);
        v15 = v23;
        v14 = v24;
        v13 = v26;
      }
    }
  }
  if ( a5 )
    v19 = sub_33FB160(v13, a3, a4, &v28, v15, v14);
  else
    v19 = sub_33FB310(v13, a3, a4, &v28, v15, v14);
  v20 = v19;
  v21 = v18;
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  v28 = a2;
  result = sub_337DC20((__int64)(a1 + 1), &v28);
  *result = v20;
  *((_DWORD *)result + 2) = v21;
  return result;
}
