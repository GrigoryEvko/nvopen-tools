// Function: sub_15C68B0
// Address: 0x15c68b0
//
__int64 __fastcall sub_15C68B0(
        __int64 *a1,
        unsigned int a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        char a7)
{
  char v11; // r11
  __int64 v12; // r9
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // eax
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  int v26; // [rsp+20h] [rbp-60h]
  int v27; // [rsp+24h] [rbp-5Ch]
  __int64 v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  unsigned __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h] BYREF
  __int64 v32[8]; // [rsp+40h] [rbp-40h] BYREF

  v11 = a7;
  if ( a6 )
  {
LABEL_4:
    v14 = *a1;
    v30 = a4;
    v31 = a5;
    v15 = v14 + 1264;
    v16 = sub_161E980(32, 2);
    v17 = v16;
    if ( v16 )
    {
      v29 = v16;
      sub_1623D80(v16, (_DWORD)a1, 29, a6, (unsigned int)&v30, 2, 0, 0);
      v17 = v29;
      *(_WORD *)(v29 + 2) = a2;
      *(_DWORD *)(v29 + 24) = a3;
    }
    return sub_15C6710(v17, a6, v15);
  }
  v12 = *a1;
  v30 = __PAIR64__(a3, a2);
  v31 = a4;
  v32[0] = a5;
  v24 = v12;
  v27 = *(_DWORD *)(v12 + 1288);
  v28 = *(_QWORD *)(v12 + 1272);
  if ( !v27 )
    goto LABEL_3;
  v23 = a5;
  v18 = sub_15B6820((int *)&v30, (int *)&v30 + 1, &v31, v32);
  a5 = v23;
  v11 = a7;
  v19 = (v27 - 1) & v18;
  v20 = (__int64 *)(v28 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == -8 )
    goto LABEL_3;
  v26 = 1;
  v22 = v24;
  while ( 1 )
  {
    if ( v21 != -16 && v30 == __PAIR64__(*(_DWORD *)(v21 + 24), *(unsigned __int16 *)(v21 + 2)) )
    {
      v25 = *(unsigned int *)(v21 + 8);
      if ( v31 == *(_QWORD *)(v21 - 8 * v25) && v32[0] == *(_QWORD *)(v21 + 8 * (1 - v25)) )
        break;
    }
    v19 = (v27 - 1) & (v26 + v19);
    v20 = (__int64 *)(v28 + 8LL * v19);
    v21 = *v20;
    if ( *v20 == -8 )
      goto LABEL_3;
    ++v26;
  }
  if ( v20 == (__int64 *)(*(_QWORD *)(v22 + 1272) + 8LL * *(unsigned int *)(v22 + 1288)) || (result = *v20) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !v11 )
      return result;
    goto LABEL_4;
  }
  return result;
}
