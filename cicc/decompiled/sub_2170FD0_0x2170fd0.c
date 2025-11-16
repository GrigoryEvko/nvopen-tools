// Function: sub_2170FD0
// Address: 0x2170fd0
//
__int64 __fastcall sub_2170FD0(__int64 a1, unsigned int a2, _QWORD *a3)
{
  __int64 v3; // rsi
  __int16 v6; // ax
  __int64 *v7; // rax
  __int64 v8; // rcx
  int v9; // edx
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // r13d
  __int64 v16; // rdi
  unsigned int v17; // r13d
  bool v18; // al
  __int64 v19; // rsi
  char v20; // di
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  char v25[8]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v26; // [rsp+18h] [rbp-28h]

  v3 = 16LL * a2;
  v6 = *(_WORD *)(a1 + 24);
  if ( v6 == 50 )
  {
    v10 = **(_QWORD **)(a1 + 32);
    if ( *(_WORD *)(v10 + 24) != 10 )
      goto LABEL_6;
    goto LABEL_15;
  }
  if ( v6 != 122 )
    goto LABEL_6;
  v7 = *(__int64 **)(a1 + 32);
  v8 = v7[5];
  v9 = *(unsigned __int16 *)(v8 + 24);
  if ( v9 != 10 && v9 != 32 )
  {
    v10 = *v7;
    if ( *(_WORD *)(v10 + 24) != 10 )
      goto LABEL_6;
LABEL_15:
    v16 = *(_QWORD *)(v10 + 88);
    v17 = *(_DWORD *)(v16 + 32);
    if ( v17 <= 0x40 )
      v18 = *(_QWORD *)(v16 + 24) == 0;
    else
      v18 = v17 == (unsigned int)sub_16A57B0(v16 + 24);
    if ( v18 )
    {
LABEL_18:
      v19 = *(_QWORD *)(a1 + 40) + v3;
      v20 = *(_BYTE *)v19;
      v21 = *(_QWORD *)(v19 + 8);
      v25[0] = v20;
      v26 = v21;
      if ( v20 )
        v22 = sub_216FFF0(v20);
      else
        v22 = sub_1F58D40((__int64)v25);
      *a3 = v22 >> 1;
      return *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
    }
LABEL_6:
    *a3 = 0;
    if ( *(__int16 *)(a1 + 24) >= 0 )
      return 0;
    if ( (unsigned int)(-*(__int16 *)(a1 + 24) - 4450) > 1 )
      return 0;
    v12 = **(_QWORD **)(a1 + 32);
    if ( *(_WORD *)(v12 + 24) != 10 )
      return 0;
    v13 = *(_QWORD *)(v12 + 88);
    v14 = *(_DWORD *)(v13 + 32);
    if ( !(v14 <= 0x40 ? *(_QWORD *)(v13 + 24) == 0 : v14 == (unsigned int)sub_16A57B0(v13 + 24)) )
      return 0;
    goto LABEL_18;
  }
  v23 = *(_QWORD *)(v8 + 88);
  v24 = *(_QWORD **)(v23 + 24);
  if ( *(_DWORD *)(v23 + 32) > 0x40u )
    v24 = (_QWORD *)*v24;
  *a3 = v24;
  return **(_QWORD **)(a1 + 32);
}
