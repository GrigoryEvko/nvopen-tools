// Function: sub_392AD60
// Address: 0x392ad60
//
__int64 __fastcall sub_392AD60(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  int v4; // eax
  _BYTE *v5; // rcx
  __int64 v6; // rdi
  bool v7; // zf
  __int64 v8; // rax

  v3 = *(_QWORD *)(a2 + 144);
  v4 = sub_392A7D0((_QWORD *)a2);
  if ( v4 == 13 || v4 == 10 )
  {
LABEL_13:
    v5 = *(_BYTE **)(a2 + 144);
    if ( v4 == 13 && v5 != (_BYTE *)(*(_QWORD *)(a2 + 152) + *(_QWORD *)(a2 + 160)) && *v5 == 10 )
      *(_QWORD *)(a2 + 144) = ++v5;
  }
  else
  {
    while ( v4 != -1 )
    {
      v4 = sub_392A7D0((_QWORD *)a2);
      if ( v4 == 10 || v4 == 13 )
        goto LABEL_13;
    }
    v5 = *(_BYTE **)(a2 + 144);
  }
  v6 = *(_QWORD *)(a2 + 120);
  if ( v6 )
  {
    (*(void (__fastcall **)(__int64, __int64, __int64, _BYTE *))(*(_QWORD *)v6 + 16LL))(v6, v3, v3, &v5[-v3 - 1]);
    v5 = *(_BYTE **)(a2 + 144);
  }
  v7 = *(_BYTE *)(a2 + 169) == 0;
  v8 = *(_QWORD *)(a2 + 104);
  *(_BYTE *)(a2 + 168) = 1;
  if ( v7 )
  {
    --v5;
    *(_BYTE *)(a2 + 169) = 1;
  }
  *(_QWORD *)(a1 + 8) = v8;
  *(_DWORD *)a1 = 9;
  *(_QWORD *)(a1 + 16) = &v5[-v8];
  *(_DWORD *)(a1 + 32) = 64;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
