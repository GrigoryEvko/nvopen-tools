// Function: sub_22088A0
// Address: 0x22088a0
//
__int64 *__fastcall sub_22088A0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // esi
  _BYTE *v6; // rax
  __int64 v7; // rax
  char *v8; // rsi
  char *v9; // rsi
  __int64 v10; // rdx
  _QWORD *v11; // rbp
  unsigned __int8 *v12; // rax
  int v13; // ebx
  unsigned __int64 v14; // r13
  __int64 v15; // rcx
  const char *v16; // rsi
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int8 *v20; // rbx
  unsigned __int8 *v21; // rax
  _BYTE *v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // [rsp+8h] [rbp-60h]
  unsigned __int64 v28; // [rsp+10h] [rbp-58h]
  __int64 v29; // [rsp+18h] [rbp-50h]
  char v30; // [rsp+27h] [rbp-41h] BYREF
  _BYTE v31[64]; // [rsp+28h] [rbp-40h] BYREF

  sub_222E2D0(&v30, a1, 0);
  if ( !v30 )
  {
    v3 = *a1;
    v4 = 4;
    goto LABEL_3;
  }
  v6 = *(_BYTE **)a2;
  *(_QWORD *)(a2 + 8) = 0;
  *v6 = 0;
  v7 = 0x3FFFFFFFFFFFFFFFLL;
  v8 = (char *)a1 + *(_QWORD *)(*a1 - 24);
  if ( *((__int64 *)v8 + 2) > 0 )
    v7 = *((_QWORD *)v8 + 2);
  v9 = v8 + 208;
  v27 = v7;
  sub_2208E20(v31, v9);
  v29 = sub_222F790(v31);
  sub_2209150(v31, v9, v10);
  v11 = *(_QWORD **)((char *)a1 + *(_QWORD *)(*a1 - 24) + 232);
  v12 = (unsigned __int8 *)v11[2];
  if ( (unsigned __int64)v12 >= v11[3] )
    v13 = (*(__int64 (__fastcall **)(_QWORD))(*v11 + 72LL))(*(__int64 *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 232));
  else
    v13 = *v12;
  v14 = 0;
  while ( 1 )
  {
    if ( v13 == -1 )
    {
      v3 = *a1;
      v4 = v14 == 0 ? 6 : 2;
      *(__int64 *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 16) = 0;
      goto LABEL_3;
    }
    v15 = *(_QWORD *)(v29 + 48);
    if ( (*(_BYTE *)(v15 + 2LL * (unsigned __int8)v13 + 1) & 0x20) != 0 )
      break;
    v16 = (const char *)v11[2];
    v17 = *(_QWORD *)(a2 + 8);
    v18 = v11[3] - (_QWORD)v16;
    if ( v18 > (__int64)(v27 - v14) )
      v18 = v27 - v14;
    if ( v18 <= 1 )
    {
      v28 = v17 + 1;
      v22 = *(_BYTE **)a2;
      if ( *(_QWORD *)a2 == a2 + 16 )
        v23 = 15;
      else
        v23 = *(_QWORD *)(a2 + 16);
      if ( v28 > v23 )
      {
        sub_2240BB0(a2, v17, 0, 0, 1);
        v22 = *(_BYTE **)a2;
      }
      v22[v17] = v13;
      ++v14;
      *(_QWORD *)(a2 + 8) = v28;
      *(_BYTE *)(*(_QWORD *)a2 + v17 + 1) = 0;
      v24 = v11[2];
      v25 = v11[3];
      if ( v24 >= v25 )
      {
        if ( (*(unsigned int (__fastcall **)(_QWORD *))(*v11 + 80LL))(v11) == -1 )
        {
          v3 = *a1;
LABEL_37:
          v4 = 2;
          *(__int64 *)((char *)a1 + *(_QWORD *)(v3 - 24) + 16) = 0;
          goto LABEL_3;
        }
        v21 = (unsigned __int8 *)v11[2];
        v25 = v11[3];
      }
      else
      {
        v21 = (unsigned __int8 *)(v24 + 1);
        v11[2] = v21;
      }
      if ( v25 <= (unsigned __int64)v21 )
      {
LABEL_33:
        v13 = (*(__int64 (__fastcall **)(_QWORD *))(*v11 + 72LL))(v11);
        goto LABEL_23;
      }
    }
    else
    {
      v19 = (unsigned __int64)&v16[v18];
      v20 = (unsigned __int8 *)(v16 + 1);
      if ( v19 > (unsigned __int64)(v16 + 1) )
      {
        do
        {
          if ( (*(_BYTE *)(v15 + 2LL * *v20 + 1) & 0x20) != 0 )
            break;
          ++v20;
        }
        while ( v19 > (unsigned __int64)v20 );
        v26 = v20 - (unsigned __int8 *)v16;
      }
      else
      {
        v26 = 1;
      }
      if ( 0x3FFFFFFFFFFFFFFFLL - v17 < v26 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(a2, v16, v26);
      v14 += v26;
      v21 = (unsigned __int8 *)(v26 + v11[2]);
      v11[2] = v21;
      if ( (unsigned __int64)v21 >= v11[3] )
        goto LABEL_33;
    }
    v13 = *v21;
LABEL_23:
    if ( v27 <= v14 )
    {
      v3 = *a1;
      if ( v13 != -1 )
      {
        *(__int64 *)((char *)a1 + *(_QWORD *)(v3 - 24) + 16) = 0;
        return a1;
      }
      goto LABEL_37;
    }
  }
  v3 = *a1;
  *(__int64 *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 16) = 0;
  if ( v14 )
    return a1;
  v4 = 4;
LABEL_3:
  sub_222DC80((char *)a1 + *(_QWORD *)(v3 - 24), *(_DWORD *)((char *)a1 + *(_QWORD *)(v3 - 24) + 32) | (unsigned int)v4);
  return a1;
}
