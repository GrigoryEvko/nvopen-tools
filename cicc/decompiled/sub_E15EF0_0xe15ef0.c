// Function: sub_E15EF0
// Address: 0xe15ef0
//
__int64 __fastcall sub_E15EF0(__int64 a1, __int64 a2)
{
  _BYTE *v4; // r15
  __int64 v5; // r12
  int v6; // eax
  int v7; // r13d
  int i; // r12d
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  char *v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // rdi
  __int64 *v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rax
  int v22; // [rsp+8h] [rbp-58h]
  int v23; // [rsp+Ch] [rbp-54h]
  _QWORD v24[10]; // [rsp+10h] [rbp-50h] BYREF

  ++*(_DWORD *)(a2 + 32);
  sub_E14360(a2, 40);
  if ( !*(_BYTE *)(a1 + 48) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    v4 = *(_BYTE **)(a1 + 16);
    v5 = *(_QWORD *)(a2 + 8);
    v23 = *(_DWORD *)(a2 + 24);
    v6 = *(_DWORD *)(a2 + 28);
    *(_QWORD *)(a2 + 24) = -1;
    v22 = v6;
    sub_E15BE0(v4, a2);
    v7 = *(_DWORD *)(a2 + 28);
    if ( v7 == -1 )
    {
      sub_E12F20((__int64 *)a2, 3u, "...");
    }
    else if ( v7 )
    {
      for ( i = 1; v7 != i; ++i )
      {
        v9 = *(_QWORD *)(a2 + 8);
        v10 = *(_QWORD *)(a2 + 16);
        v11 = *(char **)a2;
        if ( v9 + 2 > v10 )
        {
          v12 = v9 + 994;
          v13 = 2 * v10;
          if ( v12 > v13 )
            *(_QWORD *)(a2 + 16) = v12;
          else
            *(_QWORD *)(a2 + 16) = v13;
          v14 = realloc(v11);
          *(_QWORD *)a2 = v14;
          v11 = (char *)v14;
          if ( !v14 )
            abort();
          v9 = *(_QWORD *)(a2 + 8);
        }
        *(_WORD *)&v11[v9] = 8236;
        *(_QWORD *)(a2 + 8) += 2LL;
        *(_DWORD *)(a2 + 24) = i;
        (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 32LL))(v4, a2);
        if ( (v4[9] & 0xC0) != 0x40 )
          (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
      }
    }
    else
    {
      *(_QWORD *)(a2 + 8) = v5;
    }
    --*(_DWORD *)(a2 + 32);
    *(_DWORD *)(a2 + 28) = v22;
    *(_DWORD *)(a2 + 24) = v23;
    sub_E14360(a2, 41);
    goto LABEL_16;
  }
  v15 = *(_BYTE **)(a1 + 24);
  if ( v15 )
  {
    sub_E15B30(v15, a2, 3, 1u);
LABEL_16:
    v16 = sub_E12F20((__int64 *)a2, 1u, " ");
    v17 = sub_E12F20(v16, *(_QWORD *)(a1 + 32), *(const void **)(a1 + 40));
    sub_E12F20(v17, 1u, " ");
  }
  sub_E12F20((__int64 *)a2, 3u, "...");
  if ( *(_BYTE *)(a1 + 48) || *(_QWORD *)(a1 + 24) )
  {
    v18 = sub_E12F20((__int64 *)a2, 1u, " ");
    v19 = sub_E12F20(v18, *(_QWORD *)(a1 + 32), *(const void **)(a1 + 40));
    sub_E12F20(v19, 1u, " ");
    if ( *(_BYTE *)(a1 + 48) )
    {
      ++*(_DWORD *)(a2 + 32);
      sub_E14360(a2, 40);
      v20 = *(_QWORD *)(a1 + 16);
      v24[1] = 344106;
      v24[2] = v20;
      v24[0] = &unk_49DFD88;
      sub_E13D60((__int64)v24, (char **)a2);
      --*(_DWORD *)(a2 + 32);
      sub_E14360(a2, 41);
    }
    else
    {
      sub_E15B30(*(_BYTE **)(a1 + 24), a2, 3, 1u);
    }
  }
  --*(_DWORD *)(a2 + 32);
  return sub_E14360(a2, 41);
}
