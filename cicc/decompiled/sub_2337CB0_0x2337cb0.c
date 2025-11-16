// Function: sub_2337CB0
// Address: 0x2337cb0
//
__int64 __fastcall sub_2337CB0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r13
  _QWORD *v6; // r14
  unsigned __int8 v7; // al
  __int64 v9; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v10; // [rsp+8h] [rbp-D8h]
  _QWORD v11[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v12[2]; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v13[2]; // [rsp+30h] [rbp-B0h] BYREF
  char v14; // [rsp+40h] [rbp-A0h] BYREF
  int v15; // [rsp+70h] [rbp-70h]
  __int64 v16; // [rsp+78h] [rbp-68h]
  __int64 v17; // [rsp+80h] [rbp-60h]
  __int64 v18; // [rsp+88h] [rbp-58h]
  __int64 v19; // [rsp+90h] [rbp-50h]
  __int64 v20; // [rsp+98h] [rbp-48h]
  __int64 v21; // [rsp+A0h] [rbp-40h]

  v3 = *((unsigned int *)a3 + 2);
  if ( !(_DWORD)v3 )
    return 0;
  v4 = *a3;
  v5 = a1;
  v6 = a2;
  v13[0] = (unsigned __int64)&v14;
  v13[1] = 0x600000000LL;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v9 = v4 + 32 * v3;
  do
  {
    v11[0] = v5;
    v11[1] = v6;
    v12[0] = 0;
    v12[1] = 0;
    if ( !*(_QWORD *)(v4 + 16) )
      sub_4263D6(a1, a2, a3);
    a2 = v11;
    a1 = v4;
    v7 = (*(__int64 (__fastcall **)(__int64, _QWORD *, unsigned __int64 *, _QWORD *))(v4 + 24))(v4, v11, v13, v12);
    if ( v7 )
      break;
    v4 += 32;
  }
  while ( v4 != v9 );
  v10 = v7;
  sub_2337B30(v13);
  return v10;
}
