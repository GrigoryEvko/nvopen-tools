// Function: sub_FEF580
// Address: 0xfef580
//
__int64 __fastcall sub_FEF580(__int64 a1, __int64 a2)
{
  char v2; // r9
  __int64 v3; // r8
  int v4; // r10d
  unsigned int v5; // ecx
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // eax
  int v13; // r11d
  __int64 v14; // [rsp+0h] [rbp-8h]

  v2 = *(_BYTE *)(a1 + 96) & 1;
  if ( v2 )
  {
    v3 = a1 + 104;
    v4 = 3;
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 112);
    v3 = *(_QWORD *)(a1 + 104);
    if ( !(_DWORD)v10 )
      goto LABEL_13;
    v4 = v10 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 == *v6 )
    goto LABEL_4;
  v12 = 1;
  while ( v7 != -4096 )
  {
    v13 = v12 + 1;
    v5 = v4 & (v12 + v5);
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      goto LABEL_4;
    v12 = v13;
  }
  if ( v2 )
  {
    v11 = 64;
    goto LABEL_14;
  }
  v10 = *(unsigned int *)(a1 + 112);
LABEL_13:
  v11 = 16 * v10;
LABEL_14:
  v6 = (__int64 *)(v3 + v11);
LABEL_4:
  v8 = 64;
  if ( !v2 )
    v8 = 16LL * *(unsigned int *)(a1 + 112);
  if ( v6 == (__int64 *)(v3 + v8) )
  {
    BYTE4(v14) = 0;
  }
  else
  {
    BYTE4(v14) = 1;
    LODWORD(v14) = *((_DWORD *)v6 + 2);
  }
  return v14;
}
