// Function: sub_C80240
// Address: 0xc80240
//
__int64 __fastcall sub_C80240(__int64 a1, unsigned __int8 *a2, unsigned __int64 a3, unsigned int a4)
{
  unsigned __int64 v4; // rax
  unsigned __int8 v6; // r14
  bool v7; // al
  const char *v9; // r14
  size_t v10; // rax
  __int64 v11; // rcx
  size_t v12; // rdx
  int v13; // eax
  size_t v14; // rax
  unsigned __int64 v15; // [rsp+8h] [rbp-38h]
  unsigned __int8 *v16; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v17; // [rsp+18h] [rbp-28h]

  v4 = a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  v16 = a2;
  v17 = a3;
  *(_OWORD *)(a1 + 16) = 0;
  *(_OWORD *)(a1 + 32) = 0;
  if ( !a3 )
    goto LABEL_8;
  v6 = *a2;
  if ( a4 > 1 && a3 != 1 )
  {
    v15 = a3;
    v13 = isalpha(v6);
    a3 = v15;
    if ( v13 )
    {
      v4 = 2;
      if ( a2[1] == 58 )
        goto LABEL_8;
    }
  }
  if ( a3 <= 2 )
    goto LABEL_6;
  v7 = sub_C80220(v6, a4);
  v6 = *v16;
  if ( !v7 || v6 != v16[1] )
    goto LABEL_6;
  if ( sub_C80220(v16[2], a4) )
  {
    v6 = *v16;
LABEL_6:
    if ( sub_C80220(v6, a4) )
    {
      a2 = v16;
      v4 = v17 != 0;
      goto LABEL_8;
    }
    v9 = "\\/";
    if ( a4 <= 1 )
      v9 = "/";
    v10 = strlen(v9);
    v11 = 0;
    v12 = v10;
    goto LABEL_12;
  }
  v9 = "\\/";
  if ( a4 <= 1 )
    v9 = "/";
  v14 = strlen(v9);
  v11 = 2;
  v12 = v14;
LABEL_12:
  v4 = sub_C934D0(&v16, v9, v12, v11);
  a2 = v16;
  if ( v17 <= v4 )
    v4 = v17;
LABEL_8:
  *(_QWORD *)(a1 + 24) = v4;
  *(_DWORD *)(a1 + 40) = a4;
  *(_QWORD *)(a1 + 16) = a2;
  return a1;
}
