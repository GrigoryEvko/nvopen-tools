// Function: sub_15C1230
// Address: 0x15c1230
//
__int64 __fastcall sub_15C1230(__int64 *a1, __int64 a2, __int64 a3, char a4, unsigned int a5, char a6)
{
  __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r14
  int v17; // edi
  unsigned int v18; // edi
  __int64 *v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // r10
  int i; // [rsp+0h] [rbp-70h]
  char v23; // [rsp+4h] [rbp-6Ch]
  __int64 v24; // [rsp+8h] [rbp-68h]
  int v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 v28; // [rsp+28h] [rbp-48h] BYREF
  __int64 v29; // [rsp+30h] [rbp-40h]

  if ( a5 )
  {
LABEL_4:
    v13 = *a1;
    v28 = a2;
    v29 = a3;
    v27 = 0;
    v14 = v13 + 976;
    v15 = sub_161E980(32, 3);
    v16 = v15;
    if ( v15 )
    {
      sub_1623D80(v15, (_DWORD)a1, 20, a5, (unsigned int)&v27, 3, 0, 0);
      *(_WORD *)(v16 + 2) = 57;
      *(_BYTE *)(v16 + 24) = *(_BYTE *)(v16 + 24) & 0xFE | a4 & 1;
    }
    return sub_15C1090(v16, a5, v14);
  }
  v11 = *a1;
  v27 = a2;
  v28 = a3;
  LOBYTE(v29) = a4;
  v24 = v11;
  v25 = *(_DWORD *)(v11 + 1000);
  v26 = *(_QWORD *)(v11 + 984);
  if ( !v25 )
    goto LABEL_3;
  v23 = a6;
  v17 = sub_15B2420(&v27, &v28);
  a6 = v23;
  v18 = (v25 - 1) & v17;
  v19 = (__int64 *)(v26 + 8LL * v18);
  v20 = *v19;
  if ( *v19 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v20 != -16 )
    {
      v21 = *(unsigned int *)(v20 + 8);
      if ( v27 == *(_QWORD *)(v20 + 8 * (1 - v21))
        && v28 == *(_QWORD *)(v20 + 8 * (2 - v21))
        && (_BYTE)v29 == (*(_BYTE *)(v20 + 24) & 1) )
      {
        break;
      }
    }
    v18 = (v25 - 1) & (i + v18);
    v19 = (__int64 *)(v26 + 8LL * v18);
    v20 = *v19;
    if ( *v19 == -8 )
      goto LABEL_3;
  }
  if ( v19 == (__int64 *)(*(_QWORD *)(v24 + 984) + 8LL * *(unsigned int *)(v24 + 1000)) || (result = *v19) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a6 )
      return result;
    goto LABEL_4;
  }
  return result;
}
