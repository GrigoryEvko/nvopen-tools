// Function: sub_2280320
// Address: 0x2280320
//
__int64 __fastcall sub_2280320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v20; // [rsp+18h] [rbp-B8h]
  __int64 v22; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v23; // [rsp+38h] [rbp-98h] BYREF
  _QWORD v24[3]; // [rsp+40h] [rbp-90h] BYREF
  int v25; // [rsp+58h] [rbp-78h]
  char v26; // [rsp+5Ch] [rbp-74h]
  char v27; // [rsp+60h] [rbp-70h] BYREF
  __int64 v28; // [rsp+70h] [rbp-60h]
  char *v29; // [rsp+78h] [rbp-58h]
  __int64 v30; // [rsp+80h] [rbp-50h]
  int v31; // [rsp+88h] [rbp-48h]
  char v32; // [rsp+8Ch] [rbp-44h]
  char v33; // [rsp+90h] [rbp-40h] BYREF

  v5 = a3;
  v6 = *(_QWORD *)(a1 + 8);
  v22 = a3;
  if ( *(_QWORD *)a1 != v6 )
  {
    sub_22801B0(*a5, &v22);
    v8 = v22;
    v9 = v22;
    v22 = **(_QWORD **)a1;
    v10 = sub_227B160(a4, (__int64)&qword_4FDADA8, v9);
    v13 = v10;
    if ( v10 )
      v13 = *(_QWORD *)(v10 + 8);
    v26 = 1;
    v24[1] = &v27;
    v24[0] = 0;
    v24[2] = 2;
    v25 = 0;
    v28 = 0;
    v29 = &v33;
    v30 = 2;
    v31 = 0;
    v32 = 1;
    if ( !(unsigned __int8)sub_B19060((__int64)v24, (__int64)&unk_4F82400, v11, v12) )
      sub_AE6EC0((__int64)v24, (__int64)&unk_4F82420);
    sub_227AC60((__int64)v24, (__int64)&qword_4FDADA8);
    sub_227C930(a4, v8, (__int64)v24, v14);
    if ( v13 )
      sub_227F230(v22, a2, a4, v13);
    v15 = *(_QWORD *)(a1 + 8);
    v20 = *(_QWORD *)a1 + 8LL;
    while ( v15 != v20 )
    {
      v16 = *(_QWORD *)(v15 - 8);
      v17 = *a5;
      v23 = v16;
      sub_22801B0(v17, &v23);
      if ( v13 )
        sub_227F230(v16, a2, a4, v13);
      v15 -= 8;
      sub_227C930(a4, v16, (__int64)v24, v18);
    }
    v5 = v22;
    sub_227AD40((__int64)v24);
  }
  return v5;
}
