// Function: sub_164D870
// Address: 0x164d870
//
__int64 __fastcall sub_164D870(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  int v7; // ebx
  unsigned int v8; // r8d
  _QWORD *v9; // r9
  __int64 v10; // r12
  __int64 v12; // rax
  unsigned int v13; // r8d
  _QWORD *v14; // r9
  _QWORD *v15; // rcx
  void *v16; // rdi
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  void *v20; // rax
  _BYTE *v21; // rdi
  _QWORD *v22; // [rsp+0h] [rbp-160h]
  _QWORD *v23; // [rsp+8h] [rbp-158h]
  _QWORD *v24; // [rsp+8h] [rbp-158h]
  _QWORD *v25; // [rsp+8h] [rbp-158h]
  unsigned int v26; // [rsp+10h] [rbp-150h]
  _QWORD *v27; // [rsp+10h] [rbp-150h]
  unsigned int v28; // [rsp+10h] [rbp-150h]
  unsigned int v29; // [rsp+18h] [rbp-148h]
  _BYTE *v30; // [rsp+20h] [rbp-140h] BYREF
  __int64 v31; // [rsp+28h] [rbp-138h]
  _BYTE dest[304]; // [rsp+30h] [rbp-130h] BYREF

  v7 = a3;
  v8 = sub_16D19C0(a1, a2, a3);
  v9 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v8);
  if ( *v9 )
  {
    if ( *v9 != -8 )
    {
      v30 = dest;
      v31 = 0x10000000000LL;
      if ( a3 > 0x100 )
      {
        sub_16CD150(&v30, dest, a3, 1);
        v21 = &v30[(unsigned int)v31];
      }
      else
      {
        if ( !a3 )
          goto LABEL_5;
        v21 = dest;
      }
      memcpy(v21, a2, a3);
      v7 = a3 + v31;
LABEL_5:
      LODWORD(v31) = v7;
      v10 = sub_164D1F0(a1, a4, (__int64)&v30);
      if ( v30 != dest )
        _libc_free((unsigned __int64)v30);
      return v10;
    }
    --*(_DWORD *)(a1 + 16);
  }
  v23 = v9;
  v26 = v8;
  v12 = malloc(a3 + 17);
  v13 = v26;
  v14 = v23;
  v15 = (_QWORD *)v12;
  if ( !v12 )
  {
    if ( a3 == -17 )
    {
      v19 = malloc(1u);
      v13 = v26;
      v14 = v23;
      v15 = 0;
      if ( v19 )
      {
        v16 = (void *)(v19 + 16);
        v15 = (_QWORD *)v19;
        goto LABEL_19;
      }
    }
    v22 = v15;
    v25 = v14;
    v28 = v13;
    sub_16BD1C0("Allocation failed");
    v13 = v28;
    v14 = v25;
    v15 = v22;
  }
  v16 = v15 + 2;
  if ( a3 + 1 > 1 )
  {
LABEL_19:
    v24 = v15;
    v27 = v14;
    v29 = v13;
    v20 = memcpy(v16, a2, a3);
    v15 = v24;
    v14 = v27;
    v13 = v29;
    v16 = v20;
  }
  *((_BYTE *)v16 + a3) = 0;
  *v15 = a3;
  v15[1] = a4;
  *v14 = v15;
  ++*(_DWORD *)(a1 + 12);
  v17 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v13));
  v10 = *v17;
  if ( !*v17 || v10 == -8 )
  {
    v18 = v17 + 1;
    do
    {
      do
        v10 = *v18++;
      while ( !v10 );
    }
    while ( v10 == -8 );
  }
  return v10;
}
