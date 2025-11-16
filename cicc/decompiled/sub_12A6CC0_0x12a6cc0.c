// Function: sub_12A6CC0
// Address: 0x12a6cc0
//
__int64 __fastcall sub_12A6CC0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  bool v3; // r8
  _BOOL4 v4; // r15d
  unsigned __int64 v7; // r9
  _QWORD *v8; // rax
  __int64 v9; // rdi
  char v10; // r8
  _QWORD *v11; // r13
  int v12; // ecx
  __int64 v13; // rdi
  int v14; // eax
  char v16; // al
  int v17; // eax
  char v18; // [rsp+Ch] [rbp-54h]
  char *v19; // [rsp+10h] [rbp-50h] BYREF
  char v20; // [rsp+20h] [rbp-40h]
  char v21; // [rsp+21h] [rbp-3Fh]

  v3 = 0;
  v4 = 0;
  v7 = *a3;
  if ( (*(_BYTE *)(*a3 + 140) & 0xFB) == 8 )
  {
    v16 = sub_8D4C10(*a3, dword_4F077C4 != 2);
    v7 = *a3;
    v4 = (v16 & 2) != 0;
    v3 = (v16 & 2) != 0;
  }
  v18 = v3;
  v21 = 1;
  v19 = "tmp";
  v20 = 3;
  v8 = sub_127FE40(a2, v7, (__int64)&v19);
  v9 = *a3;
  v10 = v18;
  v11 = v8;
  if ( *(char *)(*a3 + 142) >= 0 && *(_BYTE *)(v9 + 140) == 12 )
  {
    v17 = sub_8D4AB0(v9);
    v10 = v18;
    v12 = v17;
  }
  else
  {
    v12 = *(_DWORD *)(v9 + 136);
  }
  sub_12A6C40(a2, (__int64)a3, v11, v12, v10);
  v13 = *a3;
  if ( *(char *)(*a3 + 142) >= 0 && *(_BYTE *)(v13 + 140) == 12 )
    v14 = sub_8D4AB0(v13);
  else
    v14 = *(_DWORD *)(v13 + 136);
  *(_QWORD *)(a1 + 8) = v11;
  *(_DWORD *)(a1 + 40) = v4;
  *(_DWORD *)(a1 + 16) = v14;
  *(_DWORD *)a1 = 0;
  return a1;
}
