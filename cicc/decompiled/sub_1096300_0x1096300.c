// Function: sub_1096300
// Address: 0x1096300
//
__int64 __fastcall sub_1096300(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  int v4; // eax
  _BYTE *v5; // rcx
  __int64 v6; // rdi
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax

  v3 = *(_QWORD *)(a2 + 152);
  v4 = sub_1095C70((_QWORD *)a2);
  if ( v4 == 13 || v4 == 10 )
  {
LABEL_13:
    v5 = *(_BYTE **)(a2 + 152);
    if ( v4 == 13 && v5 != (_BYTE *)(*(_QWORD *)(a2 + 160) + *(_QWORD *)(a2 + 168)) && *v5 == 10 )
      *(_QWORD *)(a2 + 152) = v5 + 1;
  }
  else
  {
    while ( v4 != -1 )
    {
      v4 = sub_1095C70((_QWORD *)a2);
      if ( v4 == 10 || v4 == 13 )
        goto LABEL_13;
    }
    v5 = *(_BYTE **)(a2 + 152);
  }
  v6 = *(_QWORD *)(a2 + 136);
  if ( v6 )
    (*(void (__fastcall **)(__int64, __int64, __int64, _BYTE *))(*(_QWORD *)v6 + 16LL))(v6, v3, v3, &v5[-v3 - 1]);
  v7 = *(_BYTE *)(a2 + 177) == 0;
  v8 = *(_QWORD *)(a2 + 104);
  *(_BYTE *)(a2 + 176) = 1;
  v9 = *(_QWORD *)(a2 + 152);
  if ( v7 )
  {
    --v9;
    *(_BYTE *)(a2 + 177) = 1;
  }
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = v9 - v8;
  *(_DWORD *)a1 = 9;
  *(_DWORD *)(a1 + 32) = 64;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
