// Function: sub_2B1E540
// Address: 0x2b1e540
//
__int64 __fastcall sub_2B1E540(__int64 a1, int *a2, unsigned __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v5; // r12
  int *v7; // rax
  int v8; // esi
  unsigned int v9; // r13d
  __int64 v10; // rcx
  int v11; // esi
  __int64 v12; // rdi
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // edx
  int v19; // r8d
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned int *v22; // r8
  int v24; // edx
  __int64 v25; // r13
  char v26; // al
  unsigned int v27; // esi
  __int64 v28; // rax
  int v29; // edx
  int v30; // r8d
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  bool v33; // cc
  int **v34; // [rsp+8h] [rbp-38h] BYREF
  int *v35; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v36; // [rsp+18h] [rbp-28h]

  v5 = (__int64)&a4[a5 - 1];
  v7 = *(int **)a1;
  v35 = a2;
  v36 = a3;
  v8 = *v7;
  if ( a5 != 1 )
  {
    v9 = a3;
    if ( v8 )
      goto LABEL_8;
    v10 = *a4;
    if ( !v10 )
      goto LABEL_7;
    v11 = *(_DWORD *)(v10 + 120);
    if ( v11 )
    {
      v12 = *(_QWORD *)v5;
      v13 = *(_DWORD *)(*(_QWORD *)v5 + 120LL);
      if ( v13 )
        goto LABEL_6;
    }
    else
    {
      v12 = *(_QWORD *)v5;
      v11 = *(_DWORD *)(v10 + 8);
      v13 = *(_DWORD *)(*(_QWORD *)v5 + 120LL);
      if ( v13 )
      {
LABEL_6:
        if ( v13 != v11 )
        {
LABEL_7:
          *v7 = a3;
          v8 = **(_DWORD **)a1;
          goto LABEL_8;
        }
LABEL_29:
        *v7 = v13;
        v8 = **(_DWORD **)a1;
LABEL_8:
        v14 = *(_QWORD *)(***(_QWORD ***)v5 + 8LL);
        v15 = *(unsigned __int8 *)(v14 + 8);
        if ( (_BYTE)v15 == 17 )
        {
          v8 *= *(_DWORD *)(v14 + 32);
        }
        else if ( (unsigned int)(v15 - 17) > 1 )
        {
          goto LABEL_11;
        }
        v14 = **(_QWORD **)(v14 + 16);
LABEL_11:
        v16 = sub_BCDA70((__int64 *)v14, v8);
        v17 = sub_2B097B0(*(__int64 **)(*(_QWORD *)(a1 + 8) + 3296LL), 6, v16, v35, v36, 0, 0, 0);
        v19 = v18;
        v20 = *(_QWORD *)(a1 + 16);
        if ( v19 == 1 )
          *(_DWORD *)(v20 + 8) = 1;
        if ( __OFADD__(*(_QWORD *)v20, v17) )
        {
          v33 = v17 <= 0;
          v21 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v33 )
            v21 = 0x8000000000000000LL;
        }
        else
        {
          v21 = *(_QWORD *)v20 + v17;
        }
        *(_QWORD *)v20 = v21;
        v22 = *(unsigned int **)a1;
        goto LABEL_16;
      }
    }
    v13 = *(_DWORD *)(v12 + 8);
    if ( v13 != v11 )
      goto LABEL_7;
    goto LABEL_29;
  }
  if ( !v8 )
  {
    v24 = *(_DWORD *)(*a4 + 120);
    if ( !v24 )
      v24 = *(_DWORD *)(*a4 + 8);
    *v7 = v24;
    v8 = **(_DWORD **)a1;
  }
  v25 = sub_2B08680(*(_QWORD *)(**(_QWORD **)*a4 + 8LL), v8);
  v26 = sub_B4ED80(v35, v36, **(_DWORD **)a1);
  v22 = *(unsigned int **)a1;
  if ( v26 || (v27 = *v22, v34 = &v35, sub_2B09200(&v34, v27)) )
  {
    v9 = v36;
  }
  else
  {
    v28 = sub_DFBC30(*(__int64 **)(*(_QWORD *)(a1 + 8) + 3296LL), 7, v25, (__int64)v35, v36, 0, 0, 0, 0, 0, 0);
    v30 = v29;
    v31 = *(_QWORD *)(a1 + 16);
    if ( v30 == 1 )
      *(_DWORD *)(v31 + 8) = 1;
    if ( __OFADD__(*(_QWORD *)v31, v28) )
    {
      v33 = v28 <= 0;
      v32 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v33 )
        v32 = 0x8000000000000000LL;
    }
    else
    {
      v32 = *(_QWORD *)v31 + v28;
    }
    *(_QWORD *)v31 = v32;
    v9 = v36;
    v22 = *(unsigned int **)a1;
  }
LABEL_16:
  *v22 = v9;
  return *(_QWORD *)v5;
}
