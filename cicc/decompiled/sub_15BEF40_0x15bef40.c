// Function: sub_15BEF40
// Address: 0x15bef40
//
__int64 __fastcall sub_15BEF40(__int64 *a1, int a2, char a3, __int64 a4, unsigned int a5, char a6)
{
  __int64 v11; // rcx
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 *v20; // rdx
  __int64 v21; // rdi
  char v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  int v25; // [rsp+20h] [rbp-60h]
  int v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+28h] [rbp-58h]
  __int128 v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+40h] [rbp-40h]
  __int64 v30; // [rsp+48h] [rbp-38h]

  if ( a5 )
  {
LABEL_4:
    v13 = *a1;
    v30 = a4;
    v29 = 0;
    v14 = v13 + 816;
    v28 = 0;
    v15 = sub_161E980(56, 4);
    v16 = v15;
    if ( v15 )
    {
      v27 = v15;
      sub_1623D80(v15, (_DWORD)a1, 14, a5, (unsigned int)&v28, 4, 0, 0);
      v16 = v27;
      *(_WORD *)(v27 + 2) = 21;
      *(_DWORD *)(v27 + 24) = 0;
      *(_DWORD *)(v27 + 28) = a2;
      *(_QWORD *)(v27 + 32) = 0;
      *(_DWORD *)(v27 + 48) = 0;
      *(_QWORD *)(v27 + 40) = 0;
      *(_BYTE *)(v27 + 52) = a3;
    }
    return sub_15BEDB0(v16, a5, v14);
  }
  v11 = *a1;
  LODWORD(v28) = a2;
  BYTE4(v28) = a3;
  *((_QWORD *)&v28 + 1) = a4;
  v23 = v11;
  v26 = *(_DWORD *)(v11 + 840);
  v24 = *(_QWORD *)(v11 + 824);
  if ( !v26 )
    goto LABEL_3;
  v22 = a6;
  v17 = sub_15B3730((int *)&v28, (__int8 *)&v28 + 4, (__int64 *)&v28 + 1);
  v18 = v24;
  a6 = v22;
  v19 = (v26 - 1) & v17;
  v20 = (__int64 *)(v24 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == -8 )
    goto LABEL_3;
  v25 = 1;
  while ( v21 == -16
       || (_DWORD)v28 != *(_DWORD *)(v21 + 28)
       || BYTE4(v28) != *(_BYTE *)(v21 + 52)
       || *((_QWORD *)&v28 + 1) != *(_QWORD *)(v21 + 8 * (3LL - *(unsigned int *)(v21 + 8))) )
  {
    v19 = (v26 - 1) & (v25 + v19);
    v20 = (__int64 *)(v18 + 8LL * v19);
    v21 = *v20;
    if ( *v20 == -8 )
      goto LABEL_3;
    ++v25;
  }
  if ( v20 == (__int64 *)(*(_QWORD *)(v23 + 824) + 8LL * *(unsigned int *)(v23 + 840)) || (result = *v20) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a6 )
      return result;
    goto LABEL_4;
  }
  return result;
}
