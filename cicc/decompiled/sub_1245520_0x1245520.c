// Function: sub_1245520
// Address: 0x1245520
//
__int64 __fastcall sub_1245520(__int64 a1)
{
  bool v1; // zf
  unsigned __int64 v2; // r14
  unsigned int v3; // r13d
  unsigned int v4; // r12d
  unsigned int v6; // r15d
  char v7; // [rsp+Ah] [rbp-66h] BYREF
  char v8; // [rsp+Bh] [rbp-65h] BYREF
  int v9; // [rsp+Ch] [rbp-64h] BYREF
  int v10; // [rsp+10h] [rbp-60h] BYREF
  int v11; // [rsp+14h] [rbp-5Ch] BYREF
  int v12; // [rsp+18h] [rbp-58h] BYREF
  int v13; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v14[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v15[8]; // [rsp+30h] [rbp-40h] BYREF

  v14[0] = (__int64)v15;
  v1 = *(_DWORD *)(a1 + 240) == 503;
  v14[1] = 0;
  v2 = *(_QWORD *)(a1 + 232);
  LOBYTE(v15[0]) = 0;
  v3 = *(_DWORD *)(a1 + 1224);
  if ( v1 )
  {
    v6 = *(_DWORD *)(a1 + 280);
    if ( (unsigned __int8)sub_120EA00(a1, v2, (__int64)"global", 6, (__int64)"@", 1, v3, v6) )
      goto LABEL_3;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after name") )
      goto LABEL_3;
    v3 = v6;
  }
  if ( (unsigned __int8)sub_120C500(a1, (__int64)&v9, &v7, &v10, &v11, (unsigned __int8 *)&v8)
    || (unsigned __int8)sub_120C1C0(a1, &v12)
    || (unsigned __int8)sub_120A6C0(a1, &v13) )
  {
LABEL_3:
    v4 = 1;
    goto LABEL_4;
  }
  if ( (unsigned int)(*(_DWORD *)(a1 + 240) - 98) <= 1 )
    v4 = sub_1243EC0(a1, v14, v3, v2, v9, v10, v11, v8, v12, v13);
  else
    v4 = sub_1244A70(a1, (char *)v14, v3, v2, v9, v7, v10, v11, v8, v12, v13);
LABEL_4:
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0], v15[0] + 1LL);
  return v4;
}
