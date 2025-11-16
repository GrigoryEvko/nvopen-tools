// Function: sub_C34470
// Address: 0xc34470
//
__int64 __fastcall sub_C34470(
        __int64 a1,
        void *a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 a5,
        char a6,
        _BYTE *a7)
{
  char v7; // al
  unsigned int v10; // ebx
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // r15
  unsigned int v14; // r9d
  __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // r8d
  unsigned int v18; // r15d
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // eax
  unsigned int v24; // r8d
  int v25; // eax
  __int64 v26; // rdi
  int v27; // eax
  unsigned int v28; // [rsp+0h] [rbp-40h]
  int v29; // [rsp+4h] [rbp-3Ch]
  unsigned int v30; // [rsp+4h] [rbp-3Ch]
  unsigned int v33; // [rsp+Ch] [rbp-34h]

  *a7 = 0;
  v7 = *(_BYTE *)(a1 + 20);
  if ( (v7 & 6) == 0 )
    return 1;
  v10 = (a4 + 63) >> 6;
  if ( !v10 )
    v10 = 1;
  if ( (v7 & 7) == 3 )
  {
    v18 = 0;
    sub_C45D00(a2, 0, v10);
    *a7 = ((*(_BYTE *)(a1 + 20) >> 3) ^ 1) & 1;
    return v18;
  }
  v11 = sub_C33930(a1);
  v12 = *(_DWORD *)(a1 + 16);
  v13 = v11;
  if ( v12 < 0 )
  {
    sub_C45D00(a2, 0, v10);
    v17 = *(_DWORD *)(*(_QWORD *)a1 + 8LL) - 1 - *(_DWORD *)(a1 + 16);
  }
  else
  {
    v14 = v12 + 1;
    if ( a4 < v12 + 1 )
      return 1;
    v15 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)v15 <= v14 )
    {
      v18 = 0;
      sub_C49830(a2, v10, v11, v15, 0);
      sub_C475D0(a2);
      goto LABEL_15;
    }
    v29 = v15 - v14;
    sub_C49830(a2, v10, v11, v14, (unsigned int)v15 - v14);
    v17 = v29;
  }
  if ( v17 && (v30 = v17, v28 = sub_C337D0(a1), v23 = sub_C45DF0(v13, v28), v24 = v30, v30 > v23) )
  {
    if ( v30 == v23 + 1 )
    {
      v18 = 2;
    }
    else if ( v30 > v28 << 6 || (v26 = v13, v18 = 3, v27 = sub_C45D90(v26, v30 - 1), v24 = v30, !v27) )
    {
      v18 = 1;
    }
    if ( sub_C34390(a1, a6, v18, v24) && sub_C46200(a2, 1, v10) )
      return 1;
  }
  else
  {
    v18 = 0;
  }
LABEL_15:
  v20 = sub_C45E30(a2, v10, v16);
  v21 = v20;
  v22 = v20 + 1;
  if ( (*(_BYTE *)(a1 + 20) & 8) == 0 )
  {
    if ( (a5 ^ 1) + a4 > v22 )
      goto LABEL_19;
    return 1;
  }
  if ( a5 )
  {
    if ( a4 == v22 )
    {
      v33 = v21;
      v25 = sub_C45DF0(a2, v10);
      v21 = v33;
      if ( v33 == v25 )
        goto LABEL_18;
    }
    else if ( a4 >= v22 )
    {
      goto LABEL_18;
    }
    return 1;
  }
  if ( v22 )
    return 1;
LABEL_18:
  sub_C46FB0(a2, v10, v21);
LABEL_19:
  if ( v18 )
    return 16;
  else
    *a7 = 1;
  return v18;
}
