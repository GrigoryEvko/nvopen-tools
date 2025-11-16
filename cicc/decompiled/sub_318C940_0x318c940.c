// Function: sub_318C940
// Address: 0x318c940
//
void __fastcall sub_318C940(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  char *v5; // rbx
  char *v6; // r12
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  char *v14; // r13
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rsi
  int v17; // eax
  _QWORD *v18; // rdx
  char *v19; // rbx
  _QWORD *v20; // rdi
  char *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  char *v27; // r13
  __int64 v28; // [rsp+0h] [rbp-50h] BYREF
  __int64 v29; // [rsp+8h] [rbp-48h] BYREF
  char *v30; // [rsp+10h] [rbp-40h] BYREF
  int v31; // [rsp+18h] [rbp-38h]
  char v32; // [rsp+20h] [rbp-30h] BYREF

  sub_31870B0(a1[3], a1, a3);
  sub_3186750(&v28, a1[3], (__int64)a1);
  (*(void (__fastcall **)(char **, _QWORD *))(*a1 + 80LL))(&v30, a1);
  v4 = a1[3];
  if ( *(_DWORD *)(v4 + 72) != 1 )
  {
    v5 = v30;
    v6 = &v30[8 * v31];
    if ( v30 == v6 )
      goto LABEL_5;
    do
    {
      v7 = (_QWORD *)*((_QWORD *)v6 - 1);
      v6 -= 8;
      sub_B43D60(v7);
    }
    while ( v5 != v6 );
    goto LABEL_4;
  }
  v8 = sub_22077B0(0x70u);
  v11 = v8;
  if ( v8 )
    sub_318E150(v8, &v28);
  v12 = *(unsigned int *)(v4 + 16);
  v13 = *(unsigned int *)(v4 + 20);
  v29 = v11;
  v14 = (char *)&v29;
  v15 = *(_QWORD *)(v4 + 8);
  v16 = v12 + 1;
  v17 = v12;
  if ( v12 + 1 > v13 )
  {
    v26 = v4 + 8;
    if ( v15 > (unsigned __int64)&v29 || (unsigned __int64)&v29 >= v15 + 8 * v12 )
    {
      sub_31878D0(v26, v16, v12, v15, v9, v10);
      v12 = *(unsigned int *)(v4 + 16);
      v15 = *(_QWORD *)(v4 + 8);
      v17 = *(_DWORD *)(v4 + 16);
    }
    else
    {
      v27 = (char *)&v29 - v15;
      sub_31878D0(v26, v16, v12, v15, v9, v10);
      v15 = *(_QWORD *)(v4 + 8);
      v12 = *(unsigned int *)(v4 + 16);
      v14 = &v27[v15];
      v17 = *(_DWORD *)(v4 + 16);
    }
  }
  v18 = (_QWORD *)(v15 + 8 * v12);
  if ( v18 )
  {
    *v18 = *(_QWORD *)v14;
    *(_QWORD *)v14 = 0;
    v11 = v29;
    v17 = *(_DWORD *)(v4 + 16);
  }
  *(_DWORD *)(v4 + 16) = v17 + 1;
  if ( v11 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 24LL))(v11);
  v19 = v30;
  v6 = &v30[8 * v31];
  if ( v30 != v6 )
  {
    do
    {
      v20 = *(_QWORD **)v19;
      v19 += 8;
      sub_B43D10(v20);
    }
    while ( v6 != v19 );
    v6 = v30;
    v21 = &v30[8 * v31];
    if ( v21 != v30 )
    {
      do
      {
        v22 = *(_QWORD *)v6;
        v23 = 32LL * (*(_DWORD *)(*(_QWORD *)v6 + 4LL) & 0x7FFFFFF);
        if ( (*(_BYTE *)(*(_QWORD *)v6 + 7LL) & 0x40) != 0 )
        {
          v24 = *(_QWORD *)(v22 - 8);
          v22 = v24 + v23;
        }
        else
        {
          v24 = v22 - v23;
        }
        for ( ; v22 != v24; v24 += 32 )
        {
          if ( *(_QWORD *)v24 )
          {
            v25 = *(_QWORD *)(v24 + 8);
            **(_QWORD **)(v24 + 16) = v25;
            if ( v25 )
              *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
          }
          *(_QWORD *)v24 = 0;
        }
        v6 += 8;
      }
      while ( v21 != v6 );
LABEL_4:
      v6 = v30;
    }
  }
LABEL_5:
  if ( v6 != &v32 )
    _libc_free((unsigned __int64)v6);
  if ( v28 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
}
