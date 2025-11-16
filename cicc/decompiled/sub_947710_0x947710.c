// Function: sub_947710
// Address: 0x947710
//
__int64 __fastcall sub_947710(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v5; // r14
  unsigned int v6; // r15d
  char v7; // r13
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // [rsp-10h] [rbp-B0h]
  const char *v13; // [rsp+0h] [rbp-A0h] BYREF
  char v14; // [rsp+20h] [rbp-80h]
  char v15; // [rsp+21h] [rbp-7Fh]
  unsigned int v16; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v17; // [rsp+38h] [rbp-68h]
  unsigned int v18; // [rsp+48h] [rbp-58h]
  int v19; // [rsp+60h] [rbp-40h]

  result = sub_926800((__int64)&v16, *(_QWORD *)a1, a2);
  v5 = v17;
  v6 = v18;
  v7 = v19;
  if ( v16 )
    sub_91B8A0("unexpected aggregate source type!", (_DWORD *)(a2 + 36), 1);
  v8 = *(_QWORD *)(a1 + 16);
  if ( v8 )
    goto LABEL_3;
  if ( (v19 & 1) != 0 )
  {
    v9 = *(_QWORD *)a2;
    v10 = *(_QWORD *)a1;
    v15 = 1;
    v13 = "agg.tmp";
    v14 = 3;
    v11 = sub_921D70(v10, v9, (__int64)&v13, v16);
    *(_QWORD *)(a1 + 16) = v11;
    v8 = v11;
LABEL_3:
    sub_947440(*(_QWORD *)a1, v8, *(_DWORD *)(a1 + 24), *(unsigned __int8 *)(a1 + 28), v5, v6, v7 & 1, *(_QWORD *)a2);
    return v12;
  }
  return result;
}
