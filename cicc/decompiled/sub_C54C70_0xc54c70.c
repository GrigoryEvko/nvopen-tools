// Function: sub_C54C70
// Address: 0xc54c70
//
__int64 __fastcall sub_C54C70(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  size_t v12; // rdx
  size_t v13; // r13
  const void *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  int v18; // eax
  unsigned __int8 v20; // al
  int v21; // eax
  __int64 v22; // rax
  const char *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  _QWORD v30[10]; // [rsp+0h] [rbp-50h] BYREF

  v6 = sub_CB7210(a1, a2);
  v7 = *(_QWORD *)(a2 + 24);
  v30[2] = 2;
  v8 = v6;
  v9 = *(_QWORD *)(a2 + 32);
  v30[0] = v7;
  v30[1] = v9;
  sub_C51AE0(v8, (__int64)v30);
  v10 = a1;
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( v12 )
  {
    v13 = v12;
    v14 = (const void *)v11;
    if ( (*(_BYTE *)(a2 + 13) & 4) != 0 )
    {
      v15 = sub_CB7210(a1, v30);
      v16 = sub_904010(v15, " <");
      if ( *(_QWORD *)(a2 + 64) )
      {
        v14 = *(const void **)(a2 + 56);
        v13 = *(_QWORD *)(a2 + 64);
      }
      v17 = sub_A51340(v16, v14, v13);
      sub_904010(v17, ">...");
    }
    else
    {
      v20 = *(_BYTE *)(a2 + 12);
      if ( (v20 & 0x18) != 0 )
      {
        v21 = (v20 >> 3) & 3;
      }
      else
      {
        v10 = a2;
        v21 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 8LL))(a2);
      }
      if ( v21 == 1 )
      {
        v27 = sub_CB7210(v10, v30);
        v28 = sub_904010(v27, "[=<");
        if ( *(_QWORD *)(a2 + 64) )
        {
          v14 = *(const void **)(a2 + 56);
          v13 = *(_QWORD *)(a2 + 64);
        }
        v29 = sub_A51340(v28, v14, v13);
        sub_904010(v29, ">]");
      }
      else
      {
        v22 = sub_CB7210(v10, v30);
        v23 = " <";
        if ( *(_QWORD *)(a2 + 32) != 1 )
          v23 = "=<";
        v24 = sub_904010(v22, v23);
        if ( *(_QWORD *)(a2 + 64) )
        {
          v14 = *(const void **)(a2 + 56);
          v13 = *(_QWORD *)(a2 + 64);
        }
        v25 = sub_A51340(v24, v14, v13);
        v26 = *(_BYTE **)(v25 + 32);
        if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 24) )
        {
          sub_CB5D20(v25, 62);
        }
        else
        {
          *(_QWORD *)(v25 + 32) = v26 + 1;
          *v26 = 62;
        }
      }
    }
  }
  v18 = sub_C54B90(a1, a2);
  return sub_C540D0(*(_OWORD *)(a2 + 40), a3, v18);
}
