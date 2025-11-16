// Function: sub_70CD50
// Address: 0x70cd50
//
__int64 __fastcall sub_70CD50(__int64 a1, __int64 a2, int a3, _DWORD *a4, _BYTE *a5)
{
  bool v6; // zf
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 result; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // edx
  __int16 v16[24]; // [rsp+20h] [rbp-30h] BYREF

  v6 = *(_BYTE *)(a1 + 173) == 8;
  v7 = *(_QWORD *)(a2 + 128);
  *a4 = 0;
  *a5 = 5;
  if ( !v6 )
  {
    if ( a3 && (unsigned int)sub_6210B0(a1, 0) )
    {
      *a4 = 152;
      *a5 = 5;
    }
    sub_72A510(a1, a2);
    sub_70C9E0(a2, v7, a3, v8, v9);
    if ( *(_BYTE *)(a2 + 173) != 1 )
      sub_721090(a2);
    goto LABEL_6;
  }
  if ( a3 )
  {
    *a4 = 152;
    *a5 = 5;
    sub_72A510(a1, a2);
    v13 = a3;
  }
  else
  {
    sub_72A510(a1, a2);
    v13 = 0;
  }
  result = sub_70C9E0(a2, v7, v13, v11, v12);
  if ( *(_BYTE *)(a2 + 173) == 1 )
  {
LABEL_6:
    while ( *(_BYTE *)(v7 + 140) == 12 )
      v7 = *(_QWORD *)(v7 + 160);
    sub_621EE0(v16, *(_DWORD *)(v7 + 128) * dword_4F06BA0);
    return sub_6213D0(a2 + 176, (__int64)v16);
  }
  return result;
}
