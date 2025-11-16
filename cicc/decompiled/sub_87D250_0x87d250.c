// Function: sub_87D250
// Address: 0x87d250
//
void __fastcall sub_87D250(__int64 a1, void (__fastcall **a2)(const char *, _QWORD), int a3)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  char v9[33]; // [rsp+Fh] [rbp-21h] BYREF

  v4 = sub_87D1A0(a1, v9);
  if ( !v4
    || (v5 = v4, (v6 = sub_72A270(v4, v9[0])) == 0)
    || ((*(_BYTE *)(a1 + 81) & 0x10) != 0) != ((*(_BYTE *)(v6 + 89) & 4) != 0) )
  {
LABEL_4:
    if ( !(dword_4F072C8 | a3) )
      sub_74C440((*(_BYTE *)(a1 + 81) & 0x10) != 0, *(_QWORD *)(a1 + 64), (__int64)a2);
    (*a2)(*(const char **)(*(_QWORD *)a1 + 8LL), a2);
    return;
  }
  v7 = *(_QWORD *)(a1 + 64);
  v8 = *(_QWORD *)(v6 + 40);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    if ( *(_QWORD *)(v8 + 32) != v7 )
      goto LABEL_4;
  }
  else
  {
    if ( v8 )
    {
      if ( *(_BYTE *)(v8 + 28) == 3 )
        v8 = *(_QWORD *)(v8 + 32);
      else
        v8 = 0;
    }
    if ( v8 != v7 )
      goto LABEL_4;
  }
  if ( v9[0] == 7 && (*(_BYTE *)(v5 + 172) & 1) != 0 )
  {
    (*a2)("<this-param>", a2);
  }
  else if ( a3 )
  {
    sub_74C010(v6, v9[0], (__int64)a2);
  }
  else
  {
    sub_74C550(v6, v9[0], (__int64)a2);
  }
}
