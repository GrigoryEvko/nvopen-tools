// Function: sub_14806B0
// Address: 0x14806b0
//
__int64 __fastcall sub_14806B0(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  __int64 *v8; // rax
  __int64 v9; // r8
  bool v10; // dl
  int v11; // eax
  unsigned int v12; // ebx
  unsigned int v13; // r8d
  __int64 v14; // rax
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // [rsp+0h] [rbp-50h]
  unsigned int v19; // [rsp+8h] [rbp-48h]
  bool v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-38h]

  if ( a2 != a3 )
  {
    v8 = sub_1477920(a1, a3, 1u);
    sub_158ACE0(&v21, v8);
    if ( v22 <= 0x40 )
    {
      v10 = 1LL << ((unsigned __int8)v22 - 1) != v21;
    }
    else
    {
      v9 = v21;
      v19 = v22 - 1;
      v10 = 1;
      if ( (*(_QWORD *)(v21 + 8LL * (v19 >> 6)) & (1LL << v19)) != 0 )
      {
        v18 = v21;
        v11 = sub_16A58A0(&v21);
        v9 = v18;
        v10 = v19 != v11;
      }
      if ( v9 )
      {
        v20 = v10;
        j_j___libc_free_0_0(v9);
        v10 = v20;
        if ( (a4 & 4) == 0 )
          goto LABEL_7;
        goto LABEL_11;
      }
    }
    if ( (a4 & 4) == 0 )
    {
LABEL_7:
      if ( v10 )
      {
        v12 = 0;
        v13 = 4;
        goto LABEL_12;
      }
LABEL_8:
      v12 = 0;
      v13 = 0;
LABEL_12:
      v14 = sub_1480620(a1, a3, v13);
      return sub_13A5B00(a1, a2, v14, v12, a5);
    }
LABEL_11:
    v12 = 4;
    v13 = 4;
    if ( v10 )
      goto LABEL_12;
    v17 = sub_1477BC0(a1, a2);
    v13 = 0;
    if ( v17 )
      goto LABEL_12;
    goto LABEL_8;
  }
  v16 = sub_1456040(a2);
  return sub_145CF80(a1, v16, 0, 0);
}
