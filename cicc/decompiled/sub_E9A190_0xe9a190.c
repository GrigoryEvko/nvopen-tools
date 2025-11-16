// Function: sub_E9A190
// Address: 0xe9a190
//
__int64 __fastcall sub_E9A190(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r15
  char *v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r8
  void (*v10)(); // rax
  char v11; // al
  __int64 v12; // rdi
  char *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // r14
  char v19; // al
  __int64 v20; // r9
  __int64 v21; // r13
  char v22; // al
  __int64 v23; // rcx
  char *v25; // [rsp+0h] [rbp-60h] BYREF
  __int64 v26; // [rsp+8h] [rbp-58h]
  char *v27; // [rsp+10h] [rbp-50h]
  __int16 v28; // [rsp+20h] [rbp-40h]

  v7 = (char *)a2;
  sub_E9A030(a1);
  v10 = *(void (**)())(*a1 + 120);
  if ( v10 != nullsub_98 )
    ((void (__fastcall *)(__int64 *, __int64, __int64))v10)(a1, a3, 1);
  v11 = *(_BYTE *)(a2 + 32);
  v12 = a1[1];
  if ( v11 )
  {
    if ( v11 == 1 )
    {
      v8 = 259;
      v25 = "_start";
      v28 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a2 + 33) == 1 )
      {
        v3 = *(_QWORD *)(a2 + 8);
        v13 = *(char **)a2;
      }
      else
      {
        v13 = (char *)a2;
        v11 = 2;
      }
      v8 = (__int64)"_start";
      v25 = v13;
      v26 = v3;
      v27 = "_start";
      LOBYTE(v28) = v11;
      HIBYTE(v28) = 3;
    }
  }
  else
  {
    v28 = 256;
  }
  v14 = sub_E6C380(v12, (__int64 *)&v25, 1, v8, v9);
  v17 = a1[1];
  v18 = v14;
  v19 = *(_BYTE *)(a2 + 32);
  if ( v19 )
  {
    if ( v19 == 1 )
    {
      v25 = "_end";
      v28 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a2 + 33) == 1 )
      {
        v4 = *(_QWORD *)(a2 + 8);
        v7 = *(char **)a2;
      }
      else
      {
        v19 = 2;
      }
      v25 = v7;
      v26 = v4;
      v27 = "_end";
      LOBYTE(v28) = v19;
      HIBYTE(v28) = 3;
    }
  }
  else
  {
    v28 = 256;
  }
  v21 = sub_E6C380(v17, (__int64 *)&v25, 1, v15, v16);
  v22 = *(_BYTE *)(a1[1] + 1906);
  if ( v22 )
  {
    if ( v22 != 1 )
      BUG();
    v23 = 8;
  }
  else
  {
    v23 = 4;
  }
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, _QWORD, __int64, char *, __int64, char *))(*a1 + 832))(
    a1,
    v21,
    v18,
    v23,
    *(_QWORD *)(*a1 + 832),
    v20,
    v25,
    v26,
    v27);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v18, 0);
  return v21;
}
