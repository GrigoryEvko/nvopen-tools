// Function: sub_38F4360
// Address: 0x38f4360
//
__int64 __fastcall sub_38F4360(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // eax
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // r12d
  bool v8; // r13
  bool v9; // bl
  __int64 v10; // rdx
  __int64 v11; // rsi
  int v13; // eax
  __int64 v14; // rdx
  unsigned int v15; // ecx
  int v16; // eax
  int v17; // eax
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  __int64 v19; // [rsp+8h] [rbp-58h]
  const char *v20; // [rsp+10h] [rbp-50h] BYREF
  char v21; // [rsp+20h] [rbp-40h]
  char v22; // [rsp+21h] [rbp-3Fh]

  v18 = 0;
  v19 = 0;
  v4 = sub_38F0EE0(a1, &v18, a3, a4);
  if ( (_BYTE)v4 )
    goto LABEL_15;
  v7 = v4;
  if ( v19 == 9 )
  {
    if ( *(_QWORD *)v18 != 0x6D6172665F68652ELL || (v13 = 0, *(_BYTE *)(v18 + 8) != 101) )
      v13 = 1;
    v9 = v13 == 0;
    v8 = 0;
    if ( **(_DWORD **)(a1 + 152) != 25 )
      goto LABEL_5;
    goto LABEL_11;
  }
  v8 = 0;
  v9 = 0;
  if ( v19 == 12 )
  {
    if ( *(_QWORD *)v18 != 0x665F67756265642ELL || (v16 = 0, *(_DWORD *)(v18 + 8) != 1701667186) )
      v16 = 1;
    v8 = v16 == 0;
    v9 = 0;
  }
  if ( **(_DWORD **)(a1 + 152) == 25 )
  {
LABEL_11:
    sub_38EB180(a1);
    if ( !(unsigned __int8)sub_38F0EE0(a1, &v18, v14, v15) )
    {
      if ( v19 == 9 )
      {
        if ( *(_QWORD *)v18 != 0x6D6172665F68652ELL || (v17 = 0, *(_BYTE *)(v18 + 8) != 101) )
          v17 = 1;
        v11 = 1;
        v10 = v8;
        if ( v17 )
          v11 = v9;
      }
      else
      {
        v11 = v9;
        v10 = v19 == 12 && *(_QWORD *)v18 == 0x665F67756265642ELL && *(_DWORD *)(v18 + 8) == 1701667186 || v8;
      }
      goto LABEL_6;
    }
LABEL_15:
    v22 = 1;
    v20 = "Expected an identifier";
    v21 = 3;
    return (unsigned int)sub_3909CF0(a1, &v20, 0, 0, v5, v6);
  }
LABEL_5:
  v10 = v8;
  v11 = v9;
LABEL_6:
  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 328) + 712LL))(*(_QWORD *)(a1 + 328), v11, v10);
  return v7;
}
