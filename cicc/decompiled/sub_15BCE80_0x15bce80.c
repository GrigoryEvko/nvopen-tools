// Function: sub_15BCE80
// Address: 0x15bce80
//
__int64 __fastcall sub_15BCE80(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        unsigned int a9,
        char a10)
{
  __int64 v10; // r10
  __int64 v14; // r9
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r14
  int v20; // eax
  unsigned int v21; // edi
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v25; // [rsp+18h] [rbp-88h]
  int v26; // [rsp+20h] [rbp-80h]
  __int64 v27; // [rsp+28h] [rbp-78h]
  int v28; // [rsp+30h] [rbp-70h]
  __int128 v30; // [rsp+40h] [rbp-60h] BYREF
  __int64 v31; // [rsp+50h] [rbp-50h]
  __int64 v32; // [rsp+58h] [rbp-48h]
  __int64 v33; // [rsp+60h] [rbp-40h] BYREF
  int v34; // [rsp+68h] [rbp-38h] BYREF
  int v35[13]; // [rsp+6Ch] [rbp-34h] BYREF

  v10 = a5;
  if ( a9 )
  {
LABEL_4:
    v16 = *a1;
    v31 = a3;
    v32 = a4;
    v17 = v16 + 1360;
    v33 = v10;
    v30 = 0;
    v18 = sub_161E980(56, 5);
    v19 = v18;
    if ( v18 )
    {
      sub_1623D80(v18, (_DWORD)a1, 32, a9, (unsigned int)&v30, 5, 0, 0);
      *(_WORD *)(v19 + 2) = a2;
      *(_QWORD *)(v19 + 24) = 0;
      *(_QWORD *)(v19 + 32) = a6;
      *(_QWORD *)(v19 + 40) = 0;
      *(_DWORD *)(v19 + 48) = a7;
      *(_DWORD *)(v19 + 52) = a8;
    }
    return sub_15BCCC0(v19, a9, v17);
  }
  v14 = *a1;
  LODWORD(v30) = a2;
  *((_QWORD *)&v30 + 1) = a3;
  v33 = a6;
  v31 = a4;
  v34 = a7;
  v32 = a5;
  v35[0] = a8;
  v25 = v14;
  v26 = *(_DWORD *)(v14 + 1384);
  v27 = *(_QWORD *)(v14 + 1368);
  if ( !v26 )
    goto LABEL_3;
  v20 = sub_15B4F20((int *)&v30, (__int64 *)&v30 + 1, &v33, &v34, v35);
  v10 = a5;
  v21 = (v26 - 1) & v20;
  v22 = (__int64 *)(v27 + 8LL * v21);
  v23 = *v22;
  if ( *v22 == -8 )
    goto LABEL_3;
  v28 = 1;
  while ( v23 == -16
       || (_DWORD)v30 != *(unsigned __int16 *)(v23 + 2)
       || *((_QWORD *)&v30 + 1) != *(_QWORD *)(v23 + 8 * (2LL - *(unsigned int *)(v23 + 8)))
       || v33 != *(_QWORD *)(v23 + 32)
       || v34 != *(_DWORD *)(v23 + 48)
       || v35[0] != *(_DWORD *)(v23 + 52) )
  {
    v21 = (v26 - 1) & (v28 + v21);
    v22 = (__int64 *)(v27 + 8LL * v21);
    v23 = *v22;
    if ( *v22 == -8 )
      goto LABEL_3;
    ++v28;
  }
  if ( v22 == (__int64 *)(*(_QWORD *)(v25 + 1368) + 8LL * *(unsigned int *)(v25 + 1384)) || (result = *v22) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a10 )
      return result;
    goto LABEL_4;
  }
  return result;
}
