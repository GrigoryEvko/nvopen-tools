// Function: sub_1CCAE90
// Address: 0x1ccae90
//
__int64 __fastcall sub_1CCAE90(__int64 a1, char a2)
{
  __int64 v2; // r12
  unsigned __int8 v3; // al
  _QWORD *v4; // rbx
  _QWORD *v5; // rdi
  __int64 *v7; // r12
  __int64 v8; // rdi
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-68h] BYREF
  void *s; // [rsp+10h] [rbp-60h] BYREF
  __int64 v14; // [rsp+18h] [rbp-58h]
  _QWORD *v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  int v17; // [rsp+30h] [rbp-40h]
  __int64 v18; // [rsp+38h] [rbp-38h]
  _QWORD v19[6]; // [rsp+40h] [rbp-30h] BYREF

  s = v19;
  v14 = 1;
  v15 = 0;
  v16 = 0;
  v17 = 1065353216;
  v18 = 0;
  v19[0] = 0;
  v2 = sub_1649C60(a1);
  v12 = v2;
  v3 = *(_BYTE *)(v2 + 16);
  if ( v3 == 77 )
    goto LABEL_26;
  while ( v3 <= 0x17u )
  {
    if ( !a2 || v3 != 5 || *(_WORD *)(v2 + 18) != 32 )
      goto LABEL_6;
LABEL_14:
    if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
      v7 = *(__int64 **)(v2 - 8);
    else
      v7 = (__int64 *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    v8 = *v7;
LABEL_17:
    v12 = sub_1649C60(v8);
    sub_1CCA5A0(&s, &v12, 1);
    if ( !v9 )
      goto LABEL_18;
    v2 = v12;
    v3 = *(_BYTE *)(v12 + 16);
    if ( v3 == 77 )
    {
LABEL_26:
      sub_1CCA5A0(&s, &v12, 1);
      v2 = v12;
      v3 = *(_BYTE *)(v12 + 16);
    }
  }
  if ( v3 != 78 )
  {
    if ( !a2 || v3 != 56 )
      goto LABEL_6;
    goto LABEL_14;
  }
  v10 = *(_QWORD *)(v2 - 24);
  if ( *(_BYTE *)(v10 + 16) || (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
    goto LABEL_6;
  if ( sub_1C30240(*(_DWORD *)(v10 + 36)) )
    goto LABEL_24;
  v11 = *(_QWORD *)(v2 - 24);
  if ( *(_BYTE *)(v11 + 16) )
    BUG();
  if ( *(_DWORD *)(v11 + 36) == 3660 )
  {
LABEL_24:
    v8 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    goto LABEL_17;
  }
LABEL_18:
  v2 = v12;
LABEL_6:
  v4 = v15;
  while ( v4 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    j_j___libc_free_0(v5, 16);
  }
  memset(s, 0, 8 * v14);
  v16 = 0;
  v15 = 0;
  if ( s != v19 )
    j_j___libc_free_0(s, 8 * v14);
  return v2;
}
