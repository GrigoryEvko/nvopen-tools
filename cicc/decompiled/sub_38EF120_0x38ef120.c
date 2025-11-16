// Function: sub_38EF120
// Address: 0x38ef120
//
__int64 __fastcall sub_38EF120(__int64 a1)
{
  unsigned __int64 v2; // rdx
  size_t v3; // rbx
  unsigned __int64 v4; // rax
  char v5; // r10
  unsigned __int64 v6; // rax
  size_t v7; // rbx
  unsigned __int64 v8; // rdx
  char v9; // r11
  size_t v10; // r10
  char v11; // [rsp+8h] [rbp-78h]
  char v12; // [rsp+10h] [rbp-70h]
  unsigned __int8 v13; // [rsp+1Fh] [rbp-61h]
  __int64 v14; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v15; // [rsp+30h] [rbp-50h] BYREF
  size_t v16; // [rsp+38h] [rbp-48h]
  _QWORD v17[8]; // [rsp+40h] [rbp-40h] BYREF

  v15 = v17;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  v13 = sub_38EB9C0(a1, &v14);
  if ( v13 )
  {
LABEL_2:
    v13 = 1;
    goto LABEL_3;
  }
  v2 = (unsigned __int64)v15;
  v3 = v16;
  v4 = 15;
  v5 = v14;
  if ( v15 != v17 )
    v4 = v17[0];
  if ( v16 + 1 > v4 )
  {
    v12 = v14;
    sub_2240BB0((unsigned __int64 *)&v15, v16, 0, 0, 1u);
    v2 = (unsigned __int64)v15;
    v5 = v12;
  }
  *(_BYTE *)(v2 + v3) = v5;
  v16 = v3 + 1;
  for ( *((_BYTE *)v15 + v3 + 1) = 0; **(_DWORD **)(a1 + 152) == 25; *((_BYTE *)v15 + v7 + 1) = 0 )
  {
    sub_38EB180(a1);
    if ( (unsigned __int8)sub_38EB9C0(a1, &v14) )
      goto LABEL_2;
    v6 = (unsigned __int64)v15;
    v7 = v16;
    v8 = 15;
    v9 = v14;
    if ( v15 != v17 )
      v8 = v17[0];
    v10 = v16 + 1;
    if ( v16 + 1 > v8 )
    {
      v11 = v14;
      sub_2240BB0((unsigned __int64 *)&v15, v16, 0, 0, 1u);
      v6 = (unsigned __int64)v15;
      v9 = v11;
      v10 = v7 + 1;
    }
    *(_BYTE *)(v6 + v7) = v9;
    v16 = v10;
  }
  (*(void (__fastcall **)(_QWORD, _QWORD *, size_t))(**(_QWORD **)(a1 + 328) + 816LL))(*(_QWORD *)(a1 + 328), v15, v16);
LABEL_3:
  if ( v15 != v17 )
    j_j___libc_free_0((unsigned __int64)v15);
  return v13;
}
