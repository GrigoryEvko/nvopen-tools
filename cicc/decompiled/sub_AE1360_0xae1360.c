// Function: sub_AE1360
// Address: 0xae1360
//
__int64 __fastcall sub_AE1360(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned int v4; // ebx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD v15[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-58h]
  __int64 v18; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-48h]
  __int64 v20; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+48h] [rbp-38h]

  v15[0] = a2;
  v4 = *((_DWORD *)a4 + 2);
  v15[1] = a3;
  if ( (_BYTE)a3 || !sub_CA1930(v15) )
  {
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
    {
      sub_C43690(a1, 0, 0);
      return a1;
    }
    goto LABEL_4;
  }
  v7 = sub_CA1930(v15);
  if ( v4 - 1 <= 0x3F )
  {
    v8 = 0;
    if ( v4 != 1 )
      v8 = 0xFFFFFFFFFFFFFFFFLL >> (64 - ((unsigned __int8)v4 - 1));
    if ( v7 > v8 )
    {
      *(_DWORD *)(a1 + 8) = v4;
LABEL_4:
      *(_QWORD *)a1 = 0;
      return a1;
    }
  }
  v9 = sub_CA1930(v15);
  sub_C464B0(&v16, a4, v9);
  v10 = sub_CA1930(v15);
  v19 = v17;
  if ( v17 > 0x40 )
    sub_C43780(&v18, &v16);
  else
    v18 = v16;
  sub_C47170(&v18, v10);
  v11 = v19;
  v19 = 0;
  v21 = v11;
  v20 = v18;
  sub_C46B40(a4, &v20);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  v12 = *((_DWORD *)a4 + 2);
  v13 = *a4;
  if ( v12 > 0x40 )
    v13 = *(_QWORD *)(v13 + 8LL * ((v12 - 1) >> 6));
  if ( (v13 & (1LL << ((unsigned __int8)v12 - 1))) != 0 )
  {
    sub_C46E90(&v16);
    v14 = sub_CA1930(v15);
    sub_C46A40(a4, v14);
  }
  *(_DWORD *)(a1 + 8) = v17;
  *(_QWORD *)a1 = v16;
  return a1;
}
