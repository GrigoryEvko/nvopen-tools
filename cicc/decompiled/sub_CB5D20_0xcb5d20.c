// Function: sub_CB5D20
// Address: 0xcb5d20
//
__int64 __fastcall sub_CB5D20(__int64 a1, char a2)
{
  _BYTE *v2; // rax
  _BYTE v4[4]; // [rsp+Ch] [rbp-4h] BYREF

  v2 = *(_BYTE **)(a1 + 32);
  v4[0] = a2;
  if ( (unsigned __int64)v2 < *(_QWORD *)(a1 + 24) )
    goto LABEL_2;
  if ( *(_QWORD *)(a1 + 16) )
  {
    sub_CB5AE0((__int64 *)a1);
    v2 = *(_BYTE **)(a1 + 32);
LABEL_2:
    *(_QWORD *)(a1 + 32) = v2 + 1;
    *v2 = v4[0];
    return a1;
  }
  if ( *(_DWORD *)(a1 + 44) )
  {
    sub_CB5CA0((__int64 *)a1);
    return sub_CB5D20(a1, v4[0]);
  }
  else
  {
    (*(void (__fastcall **)(__int64, _BYTE *, __int64))(*(_QWORD *)a1 + 72LL))(a1, v4, 1);
    return a1;
  }
}
