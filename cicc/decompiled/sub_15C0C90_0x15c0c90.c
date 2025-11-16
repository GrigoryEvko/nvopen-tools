// Function: sub_15C0C90
// Address: 0x15c0c90
//
__int64 __fastcall sub_15C0C90(__int64 *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, char a6)
{
  __int64 v10; // rcx
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r14
  int v16; // r11d
  unsigned int v17; // r11d
  __int64 *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // [rsp+0h] [rbp-70h]
  char v23; // [rsp+4h] [rbp-6Ch]
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  int v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  __int64 v29; // [rsp+28h] [rbp-48h] BYREF
  int v30[16]; // [rsp+30h] [rbp-40h] BYREF

  if ( a5 )
  {
LABEL_4:
    v12 = *a1;
    v28 = a3;
    v29 = a2;
    v13 = v12 + 944;
    v14 = sub_161E980(32, 2);
    v15 = v14;
    if ( v14 )
    {
      sub_1623D80(v14, (_DWORD)a1, 19, a5, (unsigned int)&v28, 2, 0, 0);
      *(_DWORD *)(v15 + 24) = a4;
      *(_WORD *)(v15 + 2) = 11;
    }
    return sub_15C0AF0(v15, a5, v13);
  }
  v10 = *a1;
  v28 = a2;
  v29 = a3;
  v30[0] = a4;
  v24 = v10;
  v26 = *(_DWORD *)(v10 + 968);
  v27 = *(_QWORD *)(v10 + 952);
  if ( !v26 )
    goto LABEL_3;
  v23 = a6;
  v16 = sub_15B2A30(&v28, &v29, v30);
  a6 = v23;
  v17 = (v26 - 1) & v16;
  v18 = (__int64 *)(v27 + 8LL * v17);
  v19 = *v18;
  if ( *v18 == -8 )
    goto LABEL_3;
  v22 = 1;
  v20 = v24;
  while ( 1 )
  {
    if ( v19 != -16 )
    {
      v21 = *(unsigned int *)(v19 + 8);
      v25 = v19;
      if ( v28 == *(_QWORD *)(v19 + 8 * (1 - v21)) )
      {
        if ( *(_BYTE *)v19 != 15 )
          v25 = *(_QWORD *)(v19 - 8 * v21);
        if ( v29 == v25 && v30[0] == *(_DWORD *)(v19 + 24) )
          break;
      }
    }
    v17 = (v26 - 1) & (v22 + v17);
    v18 = (__int64 *)(v27 + 8LL * v17);
    v19 = *v18;
    if ( *v18 == -8 )
      goto LABEL_3;
    ++v22;
  }
  if ( v18 == (__int64 *)(*(_QWORD *)(v20 + 952) + 8LL * *(unsigned int *)(v20 + 968)) || (result = *v18) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a6 )
      return result;
    goto LABEL_4;
  }
  return result;
}
