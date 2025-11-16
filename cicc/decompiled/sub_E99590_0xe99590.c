// Function: sub_E99590
// Address: 0xe99590
//
__int64 __fastcall sub_E99590(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  const char *v4; // rax
  __int64 result; // rax
  int v6; // eax
  const char *v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+20h] [rbp-10h]
  char v9; // [rsp+21h] [rbp-Fh]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(v2 + 152);
  if ( *(_DWORD *)(v3 + 336) != 4 || (v6 = *(_DWORD *)(v3 + 344), v6 == 6) || !v6 )
  {
    v9 = 1;
    v4 = ".seh_* directives are not supported on this target";
LABEL_3:
    v7 = v4;
    v8 = 3;
    sub_E66880(v2, a2, (__int64)&v7);
    return 0;
  }
  result = *(_QWORD *)(a1 + 104);
  if ( !result || *(_QWORD *)(result + 8) )
  {
    v9 = 1;
    v4 = ".seh_ directive must appear within an active frame";
    goto LABEL_3;
  }
  return result;
}
