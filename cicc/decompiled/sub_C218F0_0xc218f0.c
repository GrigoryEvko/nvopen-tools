// Function: sub_C218F0
// Address: 0xc218f0
//
__int64 __fastcall sub_C218F0(__int64 a1)
{
  __int64 v1; // rax
  _DWORD *v2; // rdx
  unsigned __int64 v3; // rax
  unsigned int v5; // ecx
  _DWORD v6[3]; // [rsp+Ch] [rbp-14h] BYREF

  v1 = *(_QWORD *)(a1 + 248);
  v2 = *(_DWORD **)(v1 + 8);
  v3 = *(_QWORD *)(v1 + 16) - (_QWORD)v2;
  if ( v3 <= 3 )
    goto LABEL_7;
  if ( *v2 == 1633968999 )
  {
    *(_QWORD *)(a1 + 208) = v2 + 1;
    *(_QWORD *)(a1 + 216) = v3 - 4;
    *(_WORD *)(a1 + 224) = 0;
  }
  else
  {
    if ( *v2 != 1734567009 )
      goto LABEL_7;
    *(_QWORD *)(a1 + 216) = v3 - 4;
    *(_QWORD *)(a1 + 208) = v2 + 1;
    *(_WORD *)(a1 + 224) = 1;
  }
  if ( !(unsigned __int8)sub_C1FD50(a1 + 208, v6) )
  {
LABEL_7:
    sub_C1AFD0();
    return 6;
  }
  if ( v6[0] != 1 )
  {
    sub_C1AFD0();
    return 2;
  }
  v5 = sub_C217F0(a1, (__int64)v6);
  if ( !v5 )
  {
    sub_C1AFD0();
    return 0;
  }
  return v5;
}
