// Function: sub_25C3E80
// Address: 0x25c3e80
//
void __fastcall sub_25C3E80(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // eax
  int v4; // eax
  __int64 v5; // rax
  int v6; // eax
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  char v11; // al
  __int64 v12; // r8
  __int64 v13; // rbx
  __int64 i; // r12
  __int64 v15; // rax
  bool v16; // dl
  __int64 v17; // rcx
  char v18; // al
  __int64 v19; // rax
  int v20; // ecx
  char v21; // al
  char v22; // al
  int v23; // esi
  int v24; // [rsp+Ch] [rbp-94h]
  __int64 v25; // [rsp+10h] [rbp-90h]
  char v26; // [rsp+18h] [rbp-88h]
  __int64 v27[2]; // [rsp+20h] [rbp-80h] BYREF
  char v28; // [rsp+30h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *(_DWORD *)(a2 + 8) & 0xFFFFFFFE;
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a1 + 8) & 0xFFFFFFFE | *(_DWORD *)(a2 + 8) & 1;
  *(_DWORD *)(a1 + 8) = v3 | *(_DWORD *)(a1 + 8) & 1;
  v4 = *(_DWORD *)(a1 + 12);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a2 + 12) = v4;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
      goto LABEL_4;
    v13 = a1 + 32;
    for ( i = a2 + 32; ; i += 96 )
    {
      v15 = *(_QWORD *)(v13 - 16);
      v16 = v15 != -8192 && v15 != -4096;
      v17 = *(_QWORD *)(i - 16);
      if ( v17 == -4096 )
      {
        *(_QWORD *)(v13 - 16) = -4096;
        *(_QWORD *)(i - 16) = v15;
        if ( v16 )
          goto LABEL_24;
      }
      else if ( v17 == -8192 )
      {
        *(_QWORD *)(v13 - 16) = -8192;
        *(_QWORD *)(i - 16) = v15;
        if ( v16 )
        {
LABEL_24:
          v21 = *(_BYTE *)(v13 - 8);
          *(_DWORD *)(i + 8) = 0;
          *(_DWORD *)(i + 12) = 2;
          *(_BYTE *)(i - 8) = v21;
          *(_QWORD *)i = i + 16;
          if ( *(_DWORD *)(v13 + 8) )
            sub_25C2C90(i, (__int64 *)v13);
          sub_25C0430(v13);
        }
      }
      else if ( v16 )
      {
        v25 = *(_QWORD *)(v13 - 16);
        v22 = *(_BYTE *)(v13 - 8);
        v23 = *(_DWORD *)(v13 + 8);
        v27[1] = 0x200000000LL;
        v26 = v22;
        v27[0] = (__int64)&v28;
        if ( v23 )
        {
          sub_25C2C90((__int64)v27, (__int64 *)v13);
          v17 = *(_QWORD *)(i - 16);
        }
        *(_QWORD *)(v13 - 16) = v17;
        *(_BYTE *)(v13 - 8) = *(_BYTE *)(i - 8);
        sub_25C2C90(v13, (__int64 *)i);
        *(_QWORD *)(i - 16) = v25;
        *(_BYTE *)(i - 8) = v26;
        sub_25C2C90(i, v27);
        sub_25C0430((__int64)v27);
      }
      else
      {
        *(_QWORD *)(v13 - 16) = v17;
        *(_QWORD *)(i - 16) = v15;
        v18 = *(_BYTE *)(i - 8);
        *(_DWORD *)(v13 + 8) = 0;
        *(_BYTE *)(v13 - 8) = v18;
        *(_QWORD *)v13 = v13 + 16;
        *(_DWORD *)(v13 + 12) = 2;
        if ( *(_DWORD *)(i + 8) )
          sub_25C2C90(v13, (__int64 *)i);
        sub_25C0430(i);
      }
      v13 += 96;
      if ( a1 + 416 == v13 )
        return;
    }
  }
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v19 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    v20 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v19;
    LODWORD(v19) = *(_DWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v20;
    *(_DWORD *)(a2 + 24) = v19;
    return;
  }
  v5 = a2;
  a2 = a1;
  v2 = v5;
LABEL_4:
  v6 = *(_DWORD *)(a2 + 24);
  *(_BYTE *)(a2 + 8) |= 1u;
  v7 = (__int64 *)(v2 + 16);
  v8 = a2 + 16;
  v9 = *(_QWORD *)(a2 + 16);
  v24 = v6;
  do
  {
    v10 = *v7;
    *(_QWORD *)v8 = *v7;
    if ( v10 != -8192 && v10 != -4096 )
    {
      v11 = *((_BYTE *)v7 + 8);
      *(_DWORD *)(v8 + 24) = 0;
      v12 = (__int64)(v7 + 2);
      *(_DWORD *)(v8 + 28) = 2;
      *(_BYTE *)(v8 + 8) = v11;
      *(_QWORD *)(v8 + 16) = v8 + 32;
      if ( *((_DWORD *)v7 + 6) )
      {
        sub_25C2C90(v8 + 16, v7 + 2);
        v12 = (__int64)(v7 + 2);
      }
      sub_25C0430(v12);
    }
    v8 += 96;
    v7 += 12;
  }
  while ( v8 != a2 + 400 );
  *(_BYTE *)(v2 + 8) &= ~1u;
  *(_QWORD *)(v2 + 16) = v9;
  *(_DWORD *)(v2 + 24) = v24;
}
