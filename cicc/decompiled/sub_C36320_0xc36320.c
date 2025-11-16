// Function: sub_C36320
// Address: 0xc36320
//
__int64 __fastcall sub_C36320(__int64 a1, char a2)
{
  _DWORD *v2; // rdx
  int v3; // eax
  unsigned int v4; // r14d
  unsigned int v5; // r13d
  char *v6; // rax
  unsigned int v7; // esi
  unsigned int v8; // r13d
  __int64 v10; // rax

  v2 = *(_DWORD **)a1;
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 16LL);
  if ( v3 == 2 )
    goto LABEL_6;
  if ( a2 != 1 && a2 != 4 )
  {
    if ( a2 == 2 )
    {
      if ( (*(_BYTE *)(a1 + 20) & 8) == 0 )
        goto LABEL_11;
    }
    else if ( a2 == 3 && (*(_BYTE *)(a1 + 20) & 8) != 0 )
    {
      goto LABEL_11;
    }
LABEL_6:
    *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 2;
    *(_DWORD *)(a1 + 16) = *v2;
    v4 = v2[2];
    v5 = sub_C337D0(a1);
    v6 = (char *)sub_C33900(a1);
    v7 = v5;
    v8 = 16;
    sub_C31E80(v6, v7, v4);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) == 1 && *(_DWORD *)(*(_QWORD *)a1 + 20LL) == 1 )
    {
      v10 = sub_C33900(a1);
      sub_C45DD0(v10, 0);
    }
    return v8;
  }
LABEL_11:
  if ( v3 == 1 )
  {
    v8 = 20;
    sub_C36070(a1, 0, (*(_BYTE *)(a1 + 20) & 8) != 0, 0);
    return v8;
  }
  *(_BYTE *)(a1 + 20) &= 0xF8u;
  return 20;
}
