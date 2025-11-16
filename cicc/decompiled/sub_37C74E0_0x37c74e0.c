// Function: sub_37C74E0
// Address: 0x37c74e0
//
__int64 __fastcall sub_37C74E0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v4; // rax
  _DWORD *v5; // rdx
  int v7; // eax
  char v8; // dl
  __int64 v10; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 48);
  v5 = (_DWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_5;
  v7 = v4 & 7;
  if ( v7 )
  {
    if ( v7 != 3 || *v5 != 1 )
      goto LABEL_5;
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v5;
  }
  sub_2E8E4C0(a2, *(_QWORD *)(a1 + 32));
  if ( !v8 )
  {
LABEL_5:
    BYTE4(v10) = 0;
    return v10;
  }
  *a4 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
  return sub_37C70E0(a1, a2);
}
