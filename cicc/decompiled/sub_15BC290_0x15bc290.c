// Function: sub_15BC290
// Address: 0x15bc290
//
__int64 __fastcall sub_15BC290(__int64 *a1, __int64 a2, unsigned __int8 a3, __int64 a4, unsigned int a5, char a6)
{
  __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 *v20; // rcx
  __int64 v21; // rdi
  char v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  int v25; // [rsp+20h] [rbp-60h]
  int v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+28h] [rbp-58h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int8 v30; // [rsp+40h] [rbp-40h]

  if ( a5 )
  {
LABEL_4:
    v13 = *a1;
    v28 = a4;
    v14 = v13 + 688;
    v15 = sub_161E980(32, 1);
    v16 = v15;
    if ( v15 )
    {
      v27 = v15;
      sub_1623D80(v15, (_DWORD)a1, 10, a5, (unsigned int)&v28, 1, 0, 0);
      v16 = v27;
      *(_WORD *)(v27 + 2) = 40;
      *(_QWORD *)(v27 + 24) = a2;
      *(_DWORD *)(v27 + 4) = a3;
    }
    return sub_15BC100(v16, a5, v14);
  }
  v11 = *a1;
  v28 = a2;
  v29 = a4;
  v30 = a3;
  v23 = v11;
  v26 = *(_DWORD *)(v11 + 712);
  v24 = *(_QWORD *)(v11 + 696);
  if ( !v26 )
    goto LABEL_3;
  v22 = a6;
  v17 = sub_15B62F0(&v28, &v29);
  v18 = v24;
  a6 = v22;
  v19 = (v26 - 1) & v17;
  v20 = (__int64 *)(v24 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == -8 )
    goto LABEL_3;
  v25 = 1;
  while ( v21 == -16
       || v28 != *(_QWORD *)(v21 + 24)
       || v30 != (*(_DWORD *)(v21 + 4) != 0)
       || v29 != *(_QWORD *)(v21 - 8LL * *(unsigned int *)(v21 + 8)) )
  {
    v19 = (v26 - 1) & (v25 + v19);
    v20 = (__int64 *)(v18 + 8LL * v19);
    v21 = *v20;
    if ( *v20 == -8 )
      goto LABEL_3;
    ++v25;
  }
  if ( v20 == (__int64 *)(*(_QWORD *)(v23 + 696) + 8LL * *(unsigned int *)(v23 + 712)) || (result = *v20) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a6 )
      return result;
    goto LABEL_4;
  }
  return result;
}
