// Function: sub_12A6560
// Address: 0x12a6560
//
__int64 __fastcall sub_12A6560(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  unsigned __int64 v8; // r14
  unsigned int v9; // r15d
  char v10; // r13
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // [rsp-10h] [rbp-90h]
  const char *v16; // [rsp+0h] [rbp-80h] BYREF
  char v17; // [rsp+10h] [rbp-70h]
  char v18; // [rsp+11h] [rbp-6Fh]
  int v19; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v20; // [rsp+28h] [rbp-58h]
  unsigned int v21; // [rsp+30h] [rbp-50h]
  int v22; // [rsp+48h] [rbp-38h]

  result = sub_1286D80((__int64)&v19, *(_QWORD **)a1, a2, a4, a5);
  v8 = v20;
  v9 = v21;
  v10 = v22;
  if ( v19 )
    sub_127B550("unexpected aggregate source type!", (_DWORD *)(a2 + 36), 1);
  v11 = *(_QWORD *)(a1 + 16);
  if ( v11 )
    goto LABEL_3;
  if ( (v22 & 1) != 0 )
  {
    v12 = *(_QWORD *)a2;
    v13 = *(_QWORD **)a1;
    v18 = 1;
    v17 = 3;
    v16 = "agg.tmp";
    v14 = sub_127FE40(v13, v12, (__int64)&v16);
    *(_QWORD *)(a1 + 16) = v14;
    v11 = (unsigned __int64)v14;
LABEL_3:
    sub_12A6300(*(__int64 **)a1, v11, *(_DWORD *)(a1 + 24), *(_BYTE *)(a1 + 28), v8, v9, v10 & 1, *(_QWORD *)a2);
    return v15;
  }
  return result;
}
