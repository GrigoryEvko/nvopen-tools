// Function: sub_20ABF20
// Address: 0x20abf20
//
bool __fastcall sub_20ABF20(_DWORD *a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rdx
  __int64 v7; // r13
  char *v9; // rdx
  char v10; // al
  __int64 v11; // rdx
  bool v12; // bl
  int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // ebx
  _QWORD *v16; // rax
  char v17; // [rsp-38h] [rbp-38h] BYREF
  __int64 v18; // [rsp-30h] [rbp-30h]

  if ( !a2 )
    return 0;
  v6 = *(unsigned __int16 *)(a2 + 24);
  if ( (_WORD)v6 == 10 || (_DWORD)v6 == 32 )
  {
    v7 = a2;
  }
  else
  {
    if ( (_WORD)v6 != 104 )
      return 0;
    v7 = sub_1D1AD70(a2, 0, v6, a4, a5, a6);
    if ( !v7 )
      return 0;
  }
  v9 = *(char **)(a2 + 40);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v17 = v10;
  v18 = v11;
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) > 0x5Fu )
    {
      v12 = (unsigned __int8)(v10 - 86) <= 0x17u || (unsigned __int8)(v10 - 8) <= 5u;
      goto LABEL_11;
    }
  }
  else
  {
    v12 = sub_1F58CD0((__int64)&v17);
    if ( !sub_1F58D20((__int64)&v17) )
    {
LABEL_11:
      if ( v12 )
        v13 = a1[16];
      else
        v13 = a1[15];
      goto LABEL_13;
    }
  }
  v13 = a1[17];
LABEL_13:
  v14 = *(_QWORD *)(v7 + 88);
  v15 = *(_DWORD *)(v14 + 32);
  if ( v13 )
  {
    if ( v15 <= 0x40 )
      return *(_QWORD *)(v14 + 24) == 0;
    else
      return v15 == (unsigned int)sub_16A57B0(v14 + 24);
  }
  else
  {
    v16 = *(_QWORD **)(v14 + 24);
    if ( v15 > 0x40 )
      return (*v16 & 1) == 0;
    else
      return ((unsigned __int8)v16 & 1) == 0;
  }
}
