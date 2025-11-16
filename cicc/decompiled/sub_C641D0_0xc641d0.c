// Function: sub_C641D0
// Address: 0xc641d0
//
void __fastcall __noreturn sub_C641D0(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // [rsp+8h] [rbp-C8h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v9[24]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v10; // [rsp+50h] [rbp-80h]
  __int64 v11[4]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v12; // [rsp+80h] [rbp-50h]
  __int64 v13; // [rsp+88h] [rbp-48h]
  _QWORD *v14; // [rsp+90h] [rbp-40h]

  v8[0] = v9;
  v13 = 0x100000000LL;
  v8[1] = 0;
  v9[0] = 0;
  memset(&v11[1], 0, 24);
  v12 = 0;
  v11[0] = (__int64)&unk_49DD210;
  v14 = v8;
  sub_CB5980(v11, 0, 0, 0);
  LOWORD(v10) = 257;
  v2 = *a1;
  *a1 = 0;
  v7 = v2 | 1;
  sub_C63F70((unsigned __int64 *)&v7, v11, v3, v4, v5, v6, v9[16]);
  if ( (v7 & 1) == 0 && (v7 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v11[0] = (__int64)&unk_49DD210;
    sub_CB5840(v11);
    v11[0] = (__int64)v8;
    LOWORD(v12) = 260;
    sub_C64D30(v11, a2);
  }
  sub_C63C30(&v7, (__int64)v11);
}
