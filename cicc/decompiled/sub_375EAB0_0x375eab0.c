// Function: sub_375EAB0
// Address: 0x375eab0
//
void __fastcall sub_375EAB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rsi
  char v5; // si
  __int64 v6; // rax
  __int64 v7; // r8
  int v8; // ecx
  unsigned int v9; // edi
  __int64 v10; // rax
  int v11; // r9d
  __int64 v12; // rax
  int v13; // eax
  int v14; // r10d
  int v15[5]; // [rsp+Ch] [rbp-14h] BYREF

  v4 = *(_QWORD *)a2;
  if ( *(_DWORD *)(v4 + 36) >= 0xFFFFFFFE )
    v4 = sub_375EBD0();
  *(_QWORD *)a2 = v4;
  if ( *(_DWORD *)(v4 + 36) == -3 )
  {
    v15[0] = sub_375D5B0(a1, v4, *(_QWORD *)(a2 + 8));
    sub_37593F0(a1, v15);
    v5 = *(_BYTE *)(a1 + 512) & 1;
    if ( v5 )
    {
      v7 = a1 + 520;
      v8 = 7;
    }
    else
    {
      v6 = *(unsigned int *)(a1 + 528);
      v7 = *(_QWORD *)(a1 + 520);
      if ( !(_DWORD)v6 )
        goto LABEL_11;
      v8 = v6 - 1;
    }
    v9 = v8 & (37 * v15[0]);
    v10 = v7 + 24LL * v9;
    v11 = *(_DWORD *)v10;
    if ( v15[0] == *(_DWORD *)v10 )
    {
LABEL_8:
      *(_QWORD *)a2 = *(_QWORD *)(v10 + 8);
      *(_DWORD *)(a2 + 8) = *(_DWORD *)(v10 + 16);
      return;
    }
    v13 = 1;
    while ( v11 != -1 )
    {
      v14 = v13 + 1;
      v9 = v8 & (v13 + v9);
      v10 = v7 + 24LL * v9;
      v11 = *(_DWORD *)v10;
      if ( v15[0] == *(_DWORD *)v10 )
        goto LABEL_8;
      v13 = v14;
    }
    if ( v5 )
    {
      v12 = 192;
      goto LABEL_12;
    }
    v6 = *(unsigned int *)(a1 + 528);
LABEL_11:
    v12 = 24 * v6;
LABEL_12:
    v10 = v7 + v12;
    goto LABEL_8;
  }
}
