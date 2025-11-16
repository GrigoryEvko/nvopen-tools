// Function: sub_127FC40
// Address: 0x127fc40
//
_QWORD *__fastcall sub_127FC40(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  unsigned int v8; // edx
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-70h]
  unsigned int v21; // [rsp+Ch] [rbp-64h]
  __int64 v22; // [rsp+18h] [rbp-58h] BYREF
  char v23[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  if ( a5 )
  {
    v20 = sub_128F980(a1, a5);
    v8 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1[7] + 56LL) + 40LL)) + 4);
    v24 = 257;
    v21 = v8;
    v9 = sub_1648A60(64, 1);
    v10 = (_QWORD *)v9;
    if ( v9 )
      sub_15F8BC0(v9, a2, v21, v20, v23, 0);
    v11 = a1[7];
    if ( v11 )
    {
      v12 = (unsigned __int64 *)a1[8];
      sub_157E9D0(v11 + 40, v10);
      v13 = v10[3];
      v14 = *v12;
      v10[4] = v12;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v14 | v13 & 7;
      *(_QWORD *)(v14 + 8) = v10 + 3;
      *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780(v10, a3);
    v15 = a1[6];
    if ( v15 )
    {
      v22 = a1[6];
      sub_1623A60(&v22, v15, 2);
      if ( v10[6] )
        sub_161E7C0(v10 + 6);
      v16 = v22;
      v10[6] = v22;
      if ( v16 )
        sub_1623210(&v22, v16, v10 + 6);
    }
  }
  else
  {
    v18 = a1[46];
    v19 = sub_1648A60(64, 1);
    v10 = (_QWORD *)v19;
    if ( v19 )
      sub_15F8BE0(v19, a2, 0, a3, v18);
  }
  sub_15F8A20(v10, a4);
  return v10;
}
