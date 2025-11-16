// Function: sub_B8CF50
// Address: 0xb8cf50
//
__int64 __fastcall sub_B8CF50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v23; // rdi
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+10h] [rbp-60h] BYREF
  __int64 v28; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+28h] [rbp-48h]
  __int64 v31; // [rsp+30h] [rbp-40h]

  v8 = sub_BCB2E0(*a1);
  v9 = sub_ACD640(v8, a4, 0);
  v12 = sub_B8C140((__int64)a1, v9, v10, v11);
  v13 = sub_ACD640(v8, a5, 0);
  v16 = sub_B8C140((__int64)a1, v13, v14, v15);
  if ( a6 )
  {
    v26 = v16;
    v17 = sub_ACD640(v8, 1, 0);
    v20 = sub_B8C140((__int64)a1, v17, v18, v19);
    v21 = *a1;
    v27 = a2;
    v30 = v26;
    v28 = a3;
    v29 = v12;
    v31 = v20;
    return sub_B9C770(v21, &v27, 5, 0, 1);
  }
  else
  {
    v23 = *a1;
    v27 = a2;
    v28 = a3;
    v29 = v12;
    v30 = v16;
    return sub_B9C770(v23, &v27, 4, 0, 1);
  }
}
