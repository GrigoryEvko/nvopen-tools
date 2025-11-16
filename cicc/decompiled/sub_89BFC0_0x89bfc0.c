// Function: sub_89BFC0
// Address: 0x89bfc0
//
__int64 __fastcall sub_89BFC0(__int64 a1, __int64 a2, int a3, FILE *a4)
{
  unsigned __int64 v5; // r15
  __int64 *i; // rbx
  __int64 v7; // rcx
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // r11
  __int64 v11; // rax
  int v12; // ecx
  int v13; // eax
  __int64 v14; // rax
  _BOOL4 v15; // eax
  bool v16; // zf
  int v17; // eax
  __int64 v18; // [rsp+10h] [rbp-40h]
  int v19; // [rsp+10h] [rbp-40h]
  int v20; // [rsp+18h] [rbp-38h]
  int v21; // [rsp+18h] [rbp-38h]
  int v22; // [rsp+1Ch] [rbp-34h]

  v5 = *(_QWORD *)(a1 + 192);
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 80) - 19) <= 2u )
    v5 = *(_QWORD *)(v5 + 24);
  v22 = 0;
  for ( i = *(__int64 **)(a2 + 64); i; i = *(__int64 **)(i[5] + 32) )
  {
    if ( (*((_BYTE *)i + 89) & 4) != 0 && !*(_QWORD *)(i[21] + 168) )
      goto LABEL_5;
    v7 = *i;
    if ( (unsigned __int8)(*(_BYTE *)(*i + 80) - 4) > 1u || *(char *)(*(_QWORD *)(v7 + 88) + 177LL) >= 0 )
      break;
    v9 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 72LL);
    if ( !(v9 | v5) )
      goto LABEL_10;
    if ( !v9 || !v5 )
      goto LABEL_26;
    v10 = **(_QWORD **)(*(_QWORD *)(v9 + 88) + 32LL);
    if ( !*(_QWORD *)v5 )
    {
      v12 = 0;
      if ( !v10 )
        goto LABEL_22;
LABEL_19:
      v21 = a3;
      v19 = v12;
      v14 = sub_892BC0(v10);
      a3 = v21;
      v12 = v19;
      v13 = *(_DWORD *)(v14 + 4);
      goto LABEL_20;
    }
    v20 = a3;
    v18 = **(_QWORD **)(*(_QWORD *)(v9 + 88) + 32LL);
    v11 = sub_892BC0(*(_QWORD *)v5);
    v10 = v18;
    a3 = v20;
    v12 = *(_DWORD *)(v11 + 4);
    v13 = 0;
    if ( v18 )
      goto LABEL_19;
LABEL_20:
    if ( v13 != v12 )
      goto LABEL_26;
    v10 = *(_QWORD *)v5;
LABEL_22:
    v15 = sub_89BD20(v10, a1, v9, a4, 0, 1, a3, 8u);
    v5 = *(_QWORD *)(v5 + 24);
    v16 = !v15;
    v17 = 1;
    if ( !v16 )
      v17 = v22;
    v22 = v17;
    if ( (*((_BYTE *)i + 89) & 4) == 0 )
    {
      if ( v5 )
        goto LABEL_26;
LABEL_10:
      result = v22 ^ 1u;
      if ( !(*(_DWORD *)(a1 + 52) | v22) )
      {
        sub_88DC10(a2, a4, a1);
        return 1;
      }
      return result;
    }
LABEL_5:
    a3 = 1;
  }
  if ( !v5 )
    goto LABEL_10;
LABEL_26:
  if ( !*(_DWORD *)(a1 + 52) )
    sub_6854C0(0x308u, a4, a2);
  return 0;
}
