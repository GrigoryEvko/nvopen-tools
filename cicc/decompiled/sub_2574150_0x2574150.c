// Function: sub_2574150
// Address: 0x2574150
//
char __fastcall sub_2574150(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // r13
  _BYTE *v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  __int64 *v7; // rdx
  char result; // al
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // [rsp-8h] [rbp-68h]
  unsigned __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]

  v3 = *a1;
  v4 = (_BYTE *)a1[2];
  v5 = a1[1];
  v6 = sub_250D2C0(a2, 0);
  v16 = v7;
  v15 = v6;
  result = sub_251C230(v3, (__int64 *)&v15, v5, 0, v4, 0, 1);
  if ( !result && *(_BYTE *)a2 == 61 )
  {
    v11 = a1[3];
    v12 = *(unsigned __int64 **)(a2 + 16);
    v16 = (__int64 *)a1[4];
    v13 = *a1;
    v15 = v11;
    v17 = v13;
    return sub_2574030(v12, 0, a1[2], v14, v9, v10, v11, v16, v13, a1[1], (_BYTE *)a1[2]);
  }
  return result;
}
