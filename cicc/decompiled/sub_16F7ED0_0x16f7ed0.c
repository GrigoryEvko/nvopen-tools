// Function: sub_16F7ED0
// Address: 0x16f7ed0
//
__int64 __fastcall sub_16F7ED0(__int64 a1, unsigned int a2, unsigned int a3, _BYTE *a4)
{
  char *v7; // rsi
  unsigned int v8; // eax
  char *v9; // rax
  const char *v10; // rcx
  char *v11; // r8
  char *v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // r8d
  char *v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  const char *v19; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+10h] [rbp-30h]
  char v21; // [rsp+11h] [rbp-2Fh]

  v7 = *(char **)(a1 + 40);
  if ( a2 > *(_DWORD *)(a1 + 60) )
  {
    do
    {
      v9 = sub_16F6410(a1, v7);
      v7 = v9;
      if ( *(char **)(a1 + 40) == v9 )
        break;
      *(_QWORD *)(a1 + 40) = v9;
      v8 = *(_DWORD *)(a1 + 60) + 1;
      *(_DWORD *)(a1 + 60) = v8;
    }
    while ( v8 < a2 );
  }
  v11 = sub_16F6380(a1, v7);
  v12 = *(char **)(a1 + 40);
  if ( v12 == v11 )
    return 1;
  v13 = *(_DWORD *)(a1 + 60);
  if ( v13 > a3 )
  {
    v14 = 1;
    if ( v13 >= a2 )
      return v14;
    v15 = *(char **)(a1 + 48);
    if ( v12 == v15 )
    {
      v21 = 1;
      v19 = "A text line is less indented than the block scalar";
      v20 = 3;
      goto LABEL_20;
    }
    if ( *v12 != 35 )
    {
      v10 = "A text line is less indented than the block scalar";
      v21 = 1;
      v19 = "A text line is less indented than the block scalar";
      v20 = 3;
      if ( v12 < v15 )
      {
LABEL_11:
        v16 = *(_QWORD *)(a1 + 344);
        if ( v16 )
        {
          v17 = sub_2241E50(a1, v7, v15, v10, 1);
          *(_DWORD *)v16 = 22;
          *(_QWORD *)(v16 + 8) = v17;
        }
        if ( !*(_BYTE *)(a1 + 74) )
          sub_16D14E0(*(__int64 **)a1, *(_QWORD *)(a1 + 40), 0, (__int64)&v19, 0, 0, 0, 0, *(_BYTE *)(a1 + 75));
        *(_BYTE *)(a1 + 74) = 1;
        return 0;
      }
LABEL_20:
      *(_QWORD *)(a1 + 40) = --v15;
      goto LABEL_11;
    }
  }
  *a4 = 1;
  return 1;
}
