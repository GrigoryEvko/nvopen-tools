// Function: sub_2100010
// Address: 0x2100010
//
__int64 __fastcall sub_2100010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  int v11; // r15d
  __int64 v14; // r10
  unsigned int v15; // edx
  unsigned __int64 v16; // rcx
  __int64 v17; // r11
  __int64 v18; // rax
  unsigned int v19; // r12d
  __int64 v20; // rsi
  __int64 v21; // r12
  __int64 *v22; // rdx
  __int64 v23; // r15
  __int64 *v24; // rdx
  __int64 v25; // rdx
  _QWORD *v26; // rdi
  _QWORD *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  _QWORD *v31; // [rsp+18h] [rbp-68h]
  unsigned __int64 v32; // [rsp+20h] [rbp-60h]
  unsigned __int64 v33; // [rsp+28h] [rbp-58h]
  unsigned __int64 v34; // [rsp+30h] [rbp-50h]
  unsigned __int64 v35; // [rsp+38h] [rbp-48h]

  v6 = *(unsigned int *)(a2 + 40);
  v35 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v34 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v33 = a3 & 0xFFFFFFFFFFFFFFF8LL | 2;
  v32 = a4 & 0xFFFFFFFFFFFFFFF8LL | 2;
  if ( !(_DWORD)v6 )
    return 1;
  v8 = 0;
  v9 = 40 * v6;
  while ( 1 )
  {
    while ( 1 )
    {
      v10 = v8 + *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)v10 )
        goto LABEL_23;
      v11 = *(_DWORD *)(v10 + 8);
      if ( !v11 || (*(_BYTE *)(v10 + 4) & 1) != 0 || (*(_BYTE *)(v10 + 4) & 2) != 0 )
        goto LABEL_23;
      if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 || (*(_DWORD *)v10 & 0xFFF00) != 0 )
        break;
      v8 += 40;
      if ( v8 == v9 )
        return 1;
    }
    if ( v11 <= 0 )
      break;
    if ( !(unsigned __int8)sub_1E69FD0(*(_QWORD **)(a1 + 24), v11) )
      return 0;
LABEL_23:
    v8 += 40;
    if ( v8 == v9 )
      return 1;
  }
  v14 = *(_QWORD *)(a1 + 32);
  v15 = v11 & 0x7FFFFFFF;
  v16 = *(unsigned int *)(v14 + 408);
  v17 = v11 & 0x7FFFFFFF;
  v18 = 8 * v17;
  if ( (v11 & 0x7FFFFFFFu) < (unsigned int)v16 )
  {
    v21 = *(_QWORD *)(*(_QWORD *)(v14 + 400) + 8LL * v15);
    if ( v21 )
      goto LABEL_16;
    v19 = v15 + 1;
    if ( (unsigned int)v16 >= v15 + 1 )
      goto LABEL_14;
  }
  else
  {
    v19 = v15 + 1;
    if ( (unsigned int)v16 >= v15 + 1 )
      goto LABEL_14;
  }
  v25 = v19;
  if ( v19 < v16 )
  {
    *(_DWORD *)(v14 + 408) = v19;
  }
  else if ( v19 > v16 )
  {
    if ( v19 > (unsigned __int64)*(unsigned int *)(v14 + 412) )
    {
      v30 = *(_QWORD *)(a1 + 32);
      sub_16CD150(v14 + 400, (const void *)(v14 + 416), v19, 8, a5, a6);
      v14 = v30;
      v17 = v11 & 0x7FFFFFFF;
      v18 = 8 * v17;
      v25 = v19;
      v16 = *(unsigned int *)(v30 + 408);
    }
    v20 = *(_QWORD *)(v14 + 400);
    v26 = (_QWORD *)(v20 + 8 * v25);
    v27 = (_QWORD *)(v20 + 8 * v16);
    v28 = *(_QWORD *)(v14 + 416);
    if ( v26 != v27 )
    {
      do
        *v27++ = v28;
      while ( v26 != v27 );
      v20 = *(_QWORD *)(v14 + 400);
    }
    *(_DWORD *)(v14 + 408) = v19;
    goto LABEL_15;
  }
LABEL_14:
  v20 = *(_QWORD *)(v14 + 400);
LABEL_15:
  v29 = v17;
  v31 = (_QWORD *)v14;
  *(_QWORD *)(v20 + v18) = sub_1DBA290(v11);
  v21 = *(_QWORD *)(v31[50] + 8 * v29);
  sub_1DBB110(v31, v21);
LABEL_16:
  v22 = (__int64 *)sub_1DB3C70((__int64 *)v21, v33);
  if ( v22 == (__int64 *)(*(_QWORD *)v21 + 24LL * *(unsigned int *)(v21 + 8)) )
    goto LABEL_23;
  if ( (*(_DWORD *)((*v22 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v22 >> 1) & 3) > (*(_DWORD *)(v35 + 24) | 1u) )
    goto LABEL_23;
  v23 = v22[2];
  if ( !v23 )
    goto LABEL_23;
  if ( v34 != v35 )
  {
    v24 = (__int64 *)sub_1DB3C70((__int64 *)v21, v32);
    if ( v24 != (__int64 *)(*(_QWORD *)v21 + 24LL * *(unsigned int *)(v21 + 8))
      && (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) <= (*(_DWORD *)(v34 + 24) | 1u)
      && v23 == v24[2] )
    {
      goto LABEL_23;
    }
  }
  return 0;
}
