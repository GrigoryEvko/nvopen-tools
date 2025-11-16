// Function: sub_F0C790
// Address: 0xf0c790
//
__int64 __fastcall sub_F0C790(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // r9d
  __int64 v6; // rax
  unsigned __int8 *v7; // rdi
  unsigned __int8 *v8; // rsi
  unsigned int v9; // r10d
  unsigned __int8 *v10; // r11
  bool v11; // r14
  int v12; // eax
  unsigned int v13; // r9d
  __int64 v15; // rax
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a3;
  if ( a2 == 1 )
  {
    if ( a3 == 1 )
      return 1;
    v15 = *(_QWORD *)(a1 + 88);
    v11 = 1;
    v10 = *(unsigned __int8 **)(v15 + 32);
    v8 = &v10[*(_QWORD *)(v15 + 40)];
    goto LABEL_3;
  }
  v6 = *(_QWORD *)(a1 + 88);
  v16[0] = a2;
  v7 = *(unsigned __int8 **)(v6 + 32);
  v8 = &v7[*(_QWORD *)(v6 + 40)];
  v11 = v8 != sub_F06B50(v7, (__int64)v8, v16);
  if ( v3 != 1 )
  {
LABEL_3:
    v16[0] = v3;
    LOBYTE(v9) = v8 != sub_F06B50(v10, (__int64)v8, v16);
  }
  if ( a2 > v3 && sub_F0C740(a1, v3) )
    return 1;
  if ( !v11 && !sub_F0C740(a1, a2) )
  {
    LOBYTE(v12) = a2 >= v13;
    v9 |= v12;
  }
  return v9;
}
