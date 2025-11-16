// Function: sub_2EC90F0
// Address: 0x2ec90f0
//
__int64 __fastcall sub_2EC90F0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  int v7; // eax
  __int64 v8; // rdi
  int v9; // r15d
  char v10; // bl
  int v11; // edx
  __int64 result; // rax
  __int64 v13; // rdi
  int v14; // eax
  __int64 v15; // rdx
  unsigned int v17; // [rsp+18h] [rbp-38h] BYREF
  int v18[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v17 = 0;
  if ( a5 )
  {
    v7 = sub_2EC8D10(a5, &v17);
    v8 = *(_QWORD *)(a1 + 16);
    v18[0] = 0;
    v9 = v7;
    v10 = sub_2FF7B70(v8) & (v7 != 0);
    if ( v10 )
    {
      v14 = sub_2EC8CA0(a4);
      v15 = *(_QWORD *)(a1 + 16);
      v18[0] = v14;
      if ( v9 - *(_DWORD *)(v15 + 292) * v14 > *(_DWORD *)(v15 + 292) )
        goto LABEL_5;
    }
    if ( a3 )
    {
LABEL_4:
      *(_BYTE *)a2 = 1;
      v10 = 0;
      goto LABEL_5;
    }
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 16);
    v18[0] = 0;
    v10 = 0;
    sub_2FF7B70(v13);
    if ( a3 )
      goto LABEL_4;
  }
  v10 = sub_2EC9080(a1, a2, a4, v10 ^ 1u, v18);
  if ( v10 )
    goto LABEL_4;
LABEL_5:
  v11 = *(_DWORD *)(a4 + 276);
  result = v17;
  if ( v17 != v11 )
  {
    if ( *(_BYTE *)(a4 + 280) && !*(_DWORD *)(a2 + 4) )
      *(_DWORD *)(a2 + 4) = v11;
    if ( v10 )
      *(_DWORD *)(a2 + 8) = result;
  }
  return result;
}
