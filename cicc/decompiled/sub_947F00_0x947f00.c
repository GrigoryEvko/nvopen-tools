// Function: sub_947F00
// Address: 0x947f00
//
__int64 __fastcall sub_947F00(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  bool v4; // r8
  _BOOL4 v5; // r15d
  unsigned __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdi
  char v11; // r8
  __int64 v12; // r13
  int v13; // ecx
  __int64 v14; // rbx
  int v15; // eax
  char v17; // al
  int v18; // eax
  char v19; // [rsp+Ch] [rbp-64h]
  char *v20; // [rsp+10h] [rbp-60h] BYREF
  char v21; // [rsp+30h] [rbp-40h]
  char v22; // [rsp+31h] [rbp-3Fh]

  v4 = 0;
  v5 = 0;
  v8 = *a3;
  if ( (*(_BYTE *)(*a3 + 140) & 0xFB) == 8 )
  {
    v17 = sub_8D4C10(*a3, dword_4F077C4 != 2);
    v8 = *a3;
    v5 = (v17 & 2) != 0;
    v4 = (v17 & 2) != 0;
  }
  v19 = v4;
  v22 = 1;
  v20 = "tmp";
  v21 = 3;
  v9 = sub_921D70(a2, v8, (__int64)&v20, a4);
  v10 = *a3;
  v11 = v19;
  v12 = v9;
  if ( *(char *)(*a3 + 142) >= 0 && *(_BYTE *)(v10 + 140) == 12 )
  {
    v18 = sub_8D4AB0(v10);
    v11 = v19;
    v13 = v18;
  }
  else
  {
    v13 = *(_DWORD *)(v10 + 136);
  }
  sub_947E80(a2, (__int64)a3, v12, v13, v11);
  v14 = *a3;
  if ( *(char *)(v14 + 142) >= 0 && *(_BYTE *)(v14 + 140) == 12 )
    v15 = sub_8D4AB0(v14);
  else
    v15 = *(_DWORD *)(v14 + 136);
  *(_QWORD *)(a1 + 8) = v12;
  *(_QWORD *)(a1 + 16) = v14;
  *(_DWORD *)(a1 + 48) = v5;
  *(_DWORD *)(a1 + 24) = v15;
  *(_DWORD *)a1 = 0;
  return a1;
}
