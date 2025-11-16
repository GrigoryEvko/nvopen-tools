// Function: sub_2FC8230
// Address: 0x2fc8230
//
_BOOL8 __fastcall sub_2FC8230(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  _BOOL4 v8; // r12d
  _QWORD v10[12]; // [rsp+0h] [rbp-5A0h] BYREF
  char v11; // [rsp+60h] [rbp-540h] BYREF
  char *v12; // [rsp+A0h] [rbp-500h]
  __int64 v13; // [rsp+A8h] [rbp-4F8h]
  char v14; // [rsp+B0h] [rbp-4F0h] BYREF
  char *v15; // [rsp+130h] [rbp-470h]
  __int64 v16; // [rsp+138h] [rbp-468h]
  char v17; // [rsp+140h] [rbp-460h] BYREF
  __int64 v18; // [rsp+440h] [rbp-160h]
  __int64 v19; // [rsp+448h] [rbp-158h]
  char *v20; // [rsp+450h] [rbp-150h]
  __int64 v21; // [rsp+458h] [rbp-148h]
  char v22; // [rsp+460h] [rbp-140h] BYREF
  _QWORD *v23; // [rsp+480h] [rbp-120h]
  __int64 v24; // [rsp+488h] [rbp-118h]
  _QWORD v25[5]; // [rsp+490h] [rbp-110h] BYREF
  char v26; // [rsp+4B8h] [rbp-E8h] BYREF
  char *v27; // [rsp+4F8h] [rbp-A8h]
  __int64 v28; // [rsp+500h] [rbp-A0h]
  char v29; // [rsp+508h] [rbp-98h] BYREF
  int v30; // [rsp+538h] [rbp-68h]
  _BYTE *v31; // [rsp+540h] [rbp-60h]
  __int64 v32; // [rsp+548h] [rbp-58h]
  _BYTE v33[48]; // [rsp+550h] [rbp-50h] BYREF
  int v34; // [rsp+580h] [rbp-20h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5025C1C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_6;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5025C1C);
  v12 = &v14;
  v13 = 0x1000000000LL;
  v16 = 0x1000000000LL;
  v25[2] = v5 + 200;
  v10[10] = &v11;
  v20 = &v22;
  v25[3] = &v26;
  v10[11] = 0x800000000LL;
  v15 = &v17;
  v21 = 0x400000000LL;
  v25[4] = 0x800000000LL;
  v27 = &v29;
  v23 = v25;
  v31 = v33;
  v28 = 0x600000000LL;
  v32 = 0x600000000LL;
  memset(v10, 0, 80);
  v18 = 0;
  v19 = 0;
  v24 = 0;
  v25[0] = 0;
  v25[1] = 1;
  v30 = 0;
  v34 = 0;
  v8 = sub_2FC4FC0((__int64)v10, a2, (__int64)v33, (__int64)v25, v6, v7);
  sub_2FBF560((__int64)v10);
  return v8;
}
