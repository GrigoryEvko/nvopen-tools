// Function: sub_1651AA0
// Address: 0x1651aa0
//
void __fastcall sub_1651AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned __int8 v6; // al
  char v7; // dl
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  _QWORD v15[2]; // [rsp+0h] [rbp-50h] BYREF
  char v16; // [rsp+10h] [rbp-40h]
  char v17; // [rsp+11h] [rbp-3Fh]

  v4 = a4;
  v6 = *(_BYTE *)(a4 + 16);
  if ( v6 > 3u )
  {
    if ( v6 == 5 )
      sub_16501E0(a1, a4);
    v11 = 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
    {
      v12 = *(_QWORD *)(v4 - 8);
      v4 = v12 + v11;
    }
    else
    {
      v12 = v4 - v11;
    }
    if ( v4 != v12 )
    {
      v13 = *(_QWORD *)v12;
      v14 = *(_BYTE *)(*(_QWORD *)v12 + 16LL);
      if ( v14 != 1 )
        goto LABEL_22;
LABEL_19:
      sub_1651AA0(a1, a2, a3, *(_QWORD *)(v13 - 24));
      while ( 1 )
      {
        v12 += 24;
        if ( v4 == v12 )
          break;
        v13 = *(_QWORD *)v12;
        v14 = *(_BYTE *)(*(_QWORD *)v12 + 16LL);
        if ( v14 == 1 )
          goto LABEL_19;
LABEL_22:
        if ( v14 <= 0x10u )
          sub_1651AA0(a1, a2, a3, v13);
      }
    }
  }
  else
  {
    if ( (*(_BYTE *)(a4 + 32) & 0xF) != 1 && !sub_15E4F60(a4) )
    {
      if ( *(_BYTE *)(v4 + 16) != 1 )
        return;
      sub_1412190(a2, v4);
      if ( v7 )
        __asm { jmp     rax }
      v17 = 1;
      v15[0] = "Aliases cannot form a cycle";
      v16 = 3;
      sub_164FF40((__int64 *)a1, (__int64)v15);
      if ( !*(_QWORD *)a1 )
        return;
LABEL_11:
      sub_164FA80((__int64 *)a1, a3);
      return;
    }
    v8 = *(_QWORD *)a1;
    v17 = 1;
    v15[0] = "Alias must point to a definition";
    v16 = 3;
    if ( !v8 )
    {
      *(_BYTE *)(a1 + 72) = 1;
      return;
    }
    sub_16E2CE0(v15, v8);
    v9 = *(_BYTE **)(v8 + 24);
    if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
    {
      sub_16E7DE0(v8, 10);
    }
    else
    {
      *(_QWORD *)(v8 + 24) = v9 + 1;
      *v9 = 10;
    }
    v10 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 72) = 1;
    if ( v10 )
      goto LABEL_11;
  }
}
