// Function: sub_16E7DE0
// Address: 0x16e7de0
//
__int64 __fastcall sub_16E7DE0(__int64 a1, char a2)
{
  _BYTE *v2; // rax
  _BYTE v4[4]; // [rsp+Ch] [rbp-4h] BYREF

  v2 = *(_BYTE **)(a1 + 24);
  v4[0] = a2;
  if ( (unsigned __int64)v2 < *(_QWORD *)(a1 + 16) )
    goto LABEL_2;
  if ( *(_QWORD *)(a1 + 8) )
  {
    sub_16E7BA0((__int64 *)a1);
    v2 = *(_BYTE **)(a1 + 24);
LABEL_2:
    *(_QWORD *)(a1 + 24) = v2 + 1;
    *v2 = v4[0];
    return a1;
  }
  if ( *(_DWORD *)(a1 + 32) )
  {
    sub_16E7D60((__int64 *)a1);
    return sub_16E7DE0(a1, v4[0]);
  }
  else
  {
    (*(void (__fastcall **)(__int64, _BYTE *, __int64))(*(_QWORD *)a1 + 56LL))(a1, v4, 1);
    return a1;
  }
}
