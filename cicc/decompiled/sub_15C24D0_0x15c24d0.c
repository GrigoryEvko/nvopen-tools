// Function: sub_15C24D0
// Address: 0x15c24d0
//
__int64 __fastcall sub_15C24D0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r14
  int v16; // eax
  __int64 v17; // r8
  unsigned int v18; // edi
  __int64 *v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  int v23; // [rsp+10h] [rbp-50h]
  int v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( a4 )
  {
LABEL_4:
    v12 = *a1;
    v26[0] = a3;
    v25 = a2;
    v13 = v12 + 1040;
    v14 = sub_161E980(24, 2);
    v15 = v14;
    if ( v14 )
    {
      sub_1623D80(v14, (_DWORD)a1, 22, a4, (unsigned int)&v25, 2, 0, 0);
      *(_WORD *)(v15 + 2) = 47;
    }
    return sub_15C2340(v15, a4, v13);
  }
  v9 = *a1;
  v25 = a2;
  v26[0] = a3;
  v21 = v9;
  v22 = *(_QWORD *)(v9 + 1048);
  v24 = *(_DWORD *)(v9 + 1064);
  if ( !v24 )
    goto LABEL_3;
  v16 = sub_15B2D00(&v25, v26);
  v17 = v22;
  v18 = (v24 - 1) & v16;
  v19 = (__int64 *)(v22 + 8LL * v18);
  v20 = *v19;
  if ( *v19 == -8 )
    goto LABEL_3;
  v23 = 1;
  while ( v20 == -16
       || v25 != *(_QWORD *)(v20 - 8LL * *(unsigned int *)(v20 + 8))
       || v26[0] != *(_QWORD *)(v20 + 8 * (1LL - *(unsigned int *)(v20 + 8))) )
  {
    v18 = (v24 - 1) & (v23 + v18);
    v19 = (__int64 *)(v17 + 8LL * v18);
    v20 = *v19;
    if ( *v19 == -8 )
      goto LABEL_3;
    ++v23;
  }
  if ( v19 == (__int64 *)(*(_QWORD *)(v21 + 1048) + 8LL * *(unsigned int *)(v21 + 1064)) || (result = *v19) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a5 )
      return result;
    goto LABEL_4;
  }
  return result;
}
