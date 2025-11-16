// Function: sub_1C15400
// Address: 0x1c15400
//
__int64 __fastcall sub_1C15400(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  void (__fastcall *v3)(__int64, char **, _QWORD); // r13
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rax
  void (__fastcall *v7)(__int64, char **, _QWORD); // rbx
  unsigned int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 (__fastcall *v11)(__int64, _QWORD **); // rax
  char *v12; // [rsp+0h] [rbp-80h] BYREF
  unsigned __int64 v13; // [rsp+8h] [rbp-78h]
  _QWORD *v14; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h]
  _QWORD v16[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h]
  __int64 v19; // [rsp+40h] [rbp-40h]
  __int64 v20; // [rsp+48h] [rbp-38h]
  int v21; // [rsp+50h] [rbp-30h]
  char **v22; // [rsp+58h] [rbp-28h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    LOBYTE(v16[0]) = 0;
    v14 = v16;
    v15 = 0;
    v21 = 1;
    v17 = &unk_49EFBE0;
    v20 = 0;
    v19 = 0;
    v18 = 0;
    v22 = (char **)&v14;
    v2 = sub_16E4080(a1);
    sub_16E57C0((__int64)a2, v2, (__int64)&v17);
    if ( v20 != v18 )
      sub_16E7BA0((__int64 *)&v17);
    v12 = *v22;
    v13 = (unsigned __int64)v22[1];
    v3 = *(void (__fastcall **)(__int64, char **, _QWORD))(*(_QWORD *)a1 + 216LL);
    v4 = sub_15C8A80(v12, v13);
    v3(a1, &v12, v4);
    result = (__int64)sub_16E7BC0((__int64 *)&v17);
    if ( v14 != v16 )
      return j_j___libc_free_0(v14, v16[0] + 1LL);
  }
  else
  {
    v6 = *(_QWORD *)a1;
    v12 = 0;
    v13 = 0;
    v7 = *(void (__fastcall **)(__int64, char **, _QWORD))(v6 + 216);
    v8 = sub_15C8A80(0, 0);
    v7(a1, &v12, v8);
    v9 = sub_16E4080(a1);
    result = sub_16E5820((__int64)v12, v13, v9, a2);
    v15 = v10;
    v14 = (_QWORD *)result;
    if ( v10 )
    {
      v11 = *(__int64 (__fastcall **)(__int64, _QWORD **))(*(_QWORD *)a1 + 232LL);
      LOWORD(v19) = 261;
      v17 = &v14;
      return v11(a1, &v17);
    }
  }
  return result;
}
