// Function: sub_267C110
// Address: 0x267c110
//
void __fastcall sub_267C110(_QWORD *a1, __int64 a2, __int8 *a3, size_t a4, _QWORD **a5)
{
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 *v9; // r15
  __int64 v10; // r8
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-550h]
  __int64 v23; // [rsp+0h] [rbp-550h]
  _QWORD v25[10]; // [rsp+10h] [rbp-540h] BYREF
  _BYTE v26[344]; // [rsp+60h] [rbp-4F0h] BYREF
  __int64 v27; // [rsp+1B8h] [rbp-398h]
  _QWORD v28[10]; // [rsp+1C0h] [rbp-390h] BYREF
  _BYTE v29[352]; // [rsp+210h] [rbp-340h] BYREF
  _QWORD v30[10]; // [rsp+370h] [rbp-1E0h] BYREF
  _BYTE v31[344]; // [rsp+3C0h] [rbp-190h] BYREF
  __int64 v32; // [rsp+518h] [rbp-38h]

  if ( !a1[549] )
    return;
  v7 = sub_B43CB0(a2);
  v8 = (__int64 *)((__int64 (__fastcall *)(_QWORD, __int64))a1[549])(a1[550], v7);
  v9 = v8;
  if ( a4 <= 2 )
  {
    v10 = *v8;
LABEL_4:
    v22 = v10;
    v11 = sub_B2BE50(v10);
    if ( sub_B6EA50(v11)
      || (v18 = sub_B2BE50(v22),
          v19 = sub_B6F970(v18),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL))(v19)) )
    {
      sub_B174A0((__int64)v30, a1[551], (__int64)a3, a4, a2);
      sub_267BDC0((__int64)v28, a5, (__int64)v30);
      v30[0] = &unk_49D9D40;
      sub_23FD590((__int64)v31);
      sub_1049740(v9, (__int64)v28);
      v28[0] = &unk_49D9D40;
      sub_23FD590((__int64)v29);
    }
    return;
  }
  if ( *(_WORD *)a3 != 19791 || (v12 = 0, a3[2] != 80) )
    v12 = 1;
  v10 = *v9;
  if ( v12 )
    goto LABEL_4;
  v23 = *v9;
  v13 = sub_B2BE50(*v9);
  if ( sub_B6EA50(v13)
    || (v20 = sub_B2BE50(v23),
        v21 = sub_B6F970(v20),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 48LL))(v21)) )
  {
    sub_B174A0((__int64)v28, a1[551], (__int64)a3, a4, a2);
    sub_267BDC0((__int64)v30, a5, (__int64)v28);
    sub_B18290((__int64)v30, " [", 2u);
    sub_B18290((__int64)v30, a3, a4);
    sub_B18290((__int64)v30, "]", 1u);
    sub_23FE290((__int64)v25, (__int64)v30, v14, v15, v16, v17);
    v27 = v32;
    v25[0] = &unk_49D9D78;
    v30[0] = &unk_49D9D40;
    sub_23FD590((__int64)v31);
    v28[0] = &unk_49D9D40;
    sub_23FD590((__int64)v29);
    sub_1049740(v9, (__int64)v25);
    v25[0] = &unk_49D9D40;
    sub_23FD590((__int64)v26);
  }
}
