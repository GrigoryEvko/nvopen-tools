// Function: sub_2A739C0
// Address: 0x2a739c0
//
bool __fastcall sub_2A739C0(__int64 **a1, char a2)
{
  __int64 *v3; // r14
  __int64 *v4; // rax
  __int64 v5; // rdi
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx

  v3 = *a1;
  v4 = a1[1];
  v5 = (*a1)[4];
  if ( *(_DWORD *)v4 )
  {
    v13 = *a1[3];
    if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
      v14 = *(__int64 **)(v13 - 8);
    else
      v14 = (__int64 *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
    v15 = sub_DD8400(v5, *v14);
    v16 = v3[4];
    v17 = (*a1)[1];
    if ( a2 )
      v6 = sub_DC5000(v16, (__int64)v15, v17, 0);
    else
      v6 = sub_DC2B70(v16, (__int64)v15, v17, 0);
    v12 = sub_DD8400((*a1)[4], *a1[2]);
  }
  else
  {
    v6 = sub_DD8400(v5, *a1[2]);
    v7 = *a1[3];
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      v8 = *(_QWORD *)(v7 - 8);
    else
      v8 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    v9 = sub_DD8400((*a1)[4], *(_QWORD *)(v8 + 32));
    v10 = v3[4];
    v11 = (*a1)[1];
    if ( a2 )
      v12 = sub_DC5000(v10, (__int64)v9, v11, 0);
    else
      v12 = sub_DC2B70(v10, (__int64)v9, v11, 0);
  }
  return *a1[4] == (_QWORD)sub_2A73740(
                             (__int64)*a1,
                             (__int64)v6,
                             (__int64)v12,
                             (unsigned int)*(unsigned __int8 *)*a1[3] - 29);
}
