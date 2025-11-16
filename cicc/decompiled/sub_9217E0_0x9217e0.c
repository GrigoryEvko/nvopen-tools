// Function: sub_9217E0
// Address: 0x9217e0
//
__int64 __fastcall sub_9217E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  int v7; // ebx
  unsigned int v8; // eax
  __int64 v9; // rax
  const char *v10; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  v3 = sub_92F410();
  v4 = *a2;
  v5 = v3;
  if ( (unsigned __int8)sub_91B6F0(*a2) )
    return v5;
  v7 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 8LL) >> 8;
  if ( (unsigned int)((__int64 (*)(void))sub_91B6E0)() == v7 )
    return v5;
  v12 = 1;
  v10 = "idxprom";
  v11 = 3;
  v8 = sub_91B6E0(v4);
  v9 = sub_BCCE00(*(_QWORD *)(a1 + 40), v8);
  return sub_921630((unsigned int **)(a1 + 48), v5, v9, 0, (__int64)&v10);
}
