// Function: sub_CA86D0
// Address: 0xca86d0
//
__int64 __fastcall sub_CA86D0(__int64 a1, unsigned int a2, unsigned int a3, _BYTE *a4)
{
  char *v7; // rsi
  unsigned int v8; // eax
  char *v9; // rax
  char *v10; // rax
  __int64 v11; // r8
  char *v12; // r15
  __int64 v13; // rdx
  __int64 result; // rax
  char *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  const char *v18; // [rsp+0h] [rbp-60h] BYREF
  char v19; // [rsp+20h] [rbp-40h]
  char v20; // [rsp+21h] [rbp-3Fh]

  v7 = *(char **)(a1 + 40);
  if ( a2 > *(_DWORD *)(a1 + 60) )
  {
    do
    {
      v9 = sub_CA60E0(a1, v7);
      v7 = v9;
      if ( *(char **)(a1 + 40) == v9 )
        break;
      *(_QWORD *)(a1 + 40) = v9;
      v8 = *(_DWORD *)(a1 + 60) + 1;
      *(_DWORD *)(a1 + 60) = v8;
    }
    while ( v8 < a2 );
  }
  v10 = sub_CA6050(a1, v7);
  v12 = *(char **)(a1 + 40);
  if ( v12 == v10 )
    return 1;
  v13 = *(unsigned int *)(a1 + 60);
  if ( (unsigned int)v13 <= a3 )
    goto LABEL_17;
  result = 1;
  if ( (unsigned int)v13 >= a2 )
    return result;
  v15 = *(char **)(a1 + 48);
  if ( v12 == v15 )
  {
    v20 = 1;
    v18 = "A text line is less indented than the block scalar";
    v19 = 3;
    goto LABEL_20;
  }
  if ( *v12 == 35 )
  {
LABEL_17:
    *a4 = 1;
    return 1;
  }
  v20 = 1;
  v18 = "A text line is less indented than the block scalar";
  v19 = 3;
  if ( v12 >= v15 )
LABEL_20:
    v12 = v15 - 1;
  v16 = *(_QWORD *)(a1 + 336);
  if ( v16 )
  {
    v17 = sub_2241E50(a1, v7, v13, "A text line is less indented than the block scalar", v11);
    *(_DWORD *)v16 = 22;
    *(_QWORD *)(v16 + 8) = v17;
  }
  if ( !*(_BYTE *)(a1 + 75) )
    sub_C91CB0(*(__int64 **)a1, (unsigned __int64)v12, 0, (__int64)&v18, 0, 0, 0, 0, *(_BYTE *)(a1 + 76));
  *(_BYTE *)(a1 + 75) = 1;
  return 0;
}
