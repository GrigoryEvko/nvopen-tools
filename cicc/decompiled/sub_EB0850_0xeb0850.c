// Function: sub_EB0850
// Address: 0xeb0850
//
__int64 __fastcall sub_EB0850(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  _QWORD *v5; // rdx
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  char v8; // r11
  __int64 v9; // r10
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  char v13; // r11
  __int64 v14; // r10
  char v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h]
  char v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v21; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+38h] [rbp-48h]
  _QWORD v23[8]; // [rsp+40h] [rbp-40h] BYREF

  v21 = v23;
  v22 = 0;
  LOBYTE(v23[0]) = 0;
  v2 = sub_EAC8B0(a1, &v20);
  if ( (_BYTE)v2 )
  {
LABEL_2:
    v3 = 1;
    goto LABEL_3;
  }
  v5 = v21;
  v6 = v22;
  v3 = v2;
  v7 = 15;
  v8 = v20;
  if ( v21 != v23 )
    v7 = v23[0];
  v9 = v22 + 1;
  if ( v22 + 1 > v7 )
  {
    v17 = v20;
    v19 = v22 + 1;
    sub_2240BB0(&v21, v22, 0, 0, 1);
    v5 = v21;
    v8 = v17;
    v9 = v19;
  }
  *((_BYTE *)v5 + v6) = v8;
  v22 = v9;
  for ( *((_BYTE *)v21 + v6 + 1) = 0; **(_DWORD **)(a1 + 48) == 26; *((_BYTE *)v21 + v11 + 1) = 0 )
  {
    sub_EABFE0(a1);
    if ( (unsigned __int8)sub_EAC8B0(a1, &v20) )
      goto LABEL_2;
    v10 = v21;
    v11 = v22;
    v12 = 15;
    v13 = v20;
    if ( v21 != v23 )
      v12 = v23[0];
    v14 = v22 + 1;
    if ( v22 + 1 > v12 )
    {
      v15 = v20;
      v16 = v22 + 1;
      v18 = v22;
      sub_2240BB0(&v21, v22, 0, 0, 1);
      v10 = v21;
      v13 = v15;
      v14 = v16;
      v11 = v18;
    }
    *((_BYTE *)v10 + v11) = v13;
    v22 = v14;
  }
  (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, __int64))(**(_QWORD **)(a1 + 232) + 968LL))(
    *(_QWORD *)(a1 + 232),
    v21,
    v22,
    a2);
LABEL_3:
  if ( v21 != v23 )
    j_j___libc_free_0(v21, v23[0] + 1LL);
  return v3;
}
