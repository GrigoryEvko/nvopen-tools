// Function: sub_3939480
// Address: 0x3939480
//
__int64 __fastcall sub_3939480(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 *v6; // r13
  char v7; // al
  __int64 v8; // rax
  int v9; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_QWORD *)(*a2 + 16) - *(_QWORD *)(*a2 + 8) > 0xFFFFFFFF )
  {
    v9 = 7;
LABEL_3:
    sub_3939440(v10, &v9);
    v2 = v10[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v2 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  if ( !(unsigned __int8)sub_3939010(*a2) )
  {
    v9 = 3;
    goto LABEL_3;
  }
  v4 = *a2;
  *a2 = 0;
  v5 = sub_22077B0(0x38u);
  v6 = (__int64 *)v5;
  if ( v5 )
  {
    *(_DWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = v4;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_DWORD *)(v5 + 48) = 0;
    *(_QWORD *)v5 = &unk_4A3EF90;
    sub_393A590(v10, v5);
  }
  else
  {
    if ( v4 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
    (*(void (__fastcall **)(__int64 *, _QWORD))(MEMORY[0] + 16LL))(v10, 0);
  }
  if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_QWORD *)a1 = v10[0] & 0xFFFFFFFFFFFFFFFELL;
    v8 = *v6;
    *(_BYTE *)(a1 + 8) |= 3u;
    (*(void (__fastcall **)(__int64 *))(v8 + 8))(v6);
  }
  else
  {
    v7 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = v6;
    *(_BYTE *)(a1 + 8) = v7 & 0xFC | 2;
  }
  return a1;
}
