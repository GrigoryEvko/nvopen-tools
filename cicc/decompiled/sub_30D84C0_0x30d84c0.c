// Function: sub_30D84C0
// Address: 0x30d84c0
//
void __fastcall sub_30D84C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  int v4; // eax
  unsigned int v5; // esi
  int v6; // ebx
  __int64 v7; // rdi
  int v8; // r14d
  __int64 v9; // r10
  __int64 v10; // r8
  unsigned int v11; // ecx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _DWORD *v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rbx
  __int64 v17; // rax
  char v18; // dl
  int v19; // eax
  int v20; // edx
  __int64 v21; // [rsp+8h] [rbp-68h] BYREF
  __int64 v22[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v23[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+30h] [rbp-40h]
  __int64 v25; // [rsp+38h] [rbp-38h]
  __int64 v26; // [rsp+40h] [rbp-30h]

  v3 = a1 + 792;
  v21 = a2;
  v4 = sub_DF94A0(*(_QWORD *)(a1 + 8));
  v5 = *(_DWORD *)(a1 + 816);
  *(_DWORD *)(a1 + 780) += v4;
  v6 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 792);
    v22[0] = 0;
LABEL_28:
    sub_30D82E0(v3, 2 * v5);
LABEL_29:
    sub_30D75E0(v3, &v21, v22);
    v7 = v21;
    v9 = v22[0];
    v20 = *(_DWORD *)(a1 + 808) + 1;
    goto LABEL_24;
  }
  v7 = v21;
  v8 = 1;
  v9 = 0;
  v10 = *(_QWORD *)(a1 + 800);
  v11 = (v5 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v12 = (_QWORD *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( v21 == *v12 )
  {
LABEL_3:
    v14 = v12 + 1;
    goto LABEL_4;
  }
  while ( v13 != -4096 )
  {
    if ( !v9 && v13 == -8192 )
      v9 = (__int64)v12;
    v11 = (v5 - 1) & (v8 + v11);
    v12 = (_QWORD *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( v21 == *v12 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = (__int64)v12;
  v19 = *(_DWORD *)(a1 + 808);
  ++*(_QWORD *)(a1 + 792);
  v20 = v19 + 1;
  v22[0] = v9;
  if ( 4 * (v19 + 1) >= 3 * v5 )
    goto LABEL_28;
  if ( v5 - *(_DWORD *)(a1 + 812) - v20 <= v5 >> 3 )
  {
    sub_30D82E0(v3, v5);
    goto LABEL_29;
  }
LABEL_24:
  *(_DWORD *)(a1 + 808) = v20;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 812);
  *(_QWORD *)v9 = v7;
  v14 = (_DWORD *)(v9 + 8);
  *(_DWORD *)(v9 + 8) = 0;
LABEL_4:
  *v14 = v6;
  v15 = sub_B43CA0(v21);
  v22[0] = (__int64)v23;
  v16 = (_QWORD *)v15;
  sub_30D1540(v22, *(_BYTE **)(v15 + 232), *(_QWORD *)(v15 + 232) + *(_QWORD *)(v15 + 240));
  v24 = v16[33];
  v25 = v16[34];
  v26 = v16[35];
  if ( (unsigned int)(v24 - 42) > 1 )
  {
    if ( (_QWORD *)v22[0] != v23 )
      j_j___libc_free_0(v22[0]);
  }
  else
  {
    if ( (_QWORD *)v22[0] != v23 )
      j_j___libc_free_0(v22[0]);
    v17 = *(_QWORD *)(v21 + 72);
    v18 = *(_BYTE *)(v17 + 8);
    if ( v18 == 16 )
    {
      if ( *(_QWORD *)(v17 + 32) <= 1u )
        return;
LABEL_11:
      sub_30D0F50(a1, -500);
      return;
    }
    if ( v18 == 15 )
      goto LABEL_11;
  }
}
