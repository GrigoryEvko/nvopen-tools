// Function: sub_6F6D20
// Address: 0x6f6d20
//
__int64 __fastcall sub_6F6D20(__int64 a1, _DWORD *a2)
{
  __int64 *v2; // r14
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 i; // rax

  if ( !a1 )
    return 0;
  v2 = (__int64 *)a1;
  v3 = 0;
  v4 = 0;
  if ( (_DWORD)a2 && *(_BYTE *)(a1 + 8) == 2 )
    goto LABEL_10;
LABEL_4:
  v5 = sub_6F6C90((__int64)v2, a2);
  v6 = v5;
  if ( v4 )
    *(_QWORD *)(v4 + 16) = v5;
  else
    v3 = v5;
  for ( i = *v2; *v2; v6 = v4 )
  {
    if ( *(_BYTE *)(i + 8) == 3 )
    {
      i = sub_6BBB10(v2);
      if ( !i )
        break;
    }
    v2 = (__int64 *)i;
    v4 = v6;
    if ( !(_DWORD)a2 || *(_BYTE *)(i + 8) != 2 )
      goto LABEL_4;
LABEL_10:
    i = *v2;
  }
  return v3;
}
