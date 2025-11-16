// Function: sub_38DD280
// Address: 0x38dd280
//
__int64 __fastcall sub_38DD280(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  const char *v4; // rax
  __int64 result; // rax
  int v6; // eax
  const char *v7; // [rsp+0h] [rbp-20h] BYREF
  char v8; // [rsp+10h] [rbp-10h]
  char v9; // [rsp+11h] [rbp-Fh]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(v2 + 16);
  if ( *(_DWORD *)(v3 + 348) != 4 || (v6 = *(_DWORD *)(v3 + 352), v6 == 6) || !v6 )
  {
    v9 = 1;
    v4 = ".seh_* directives are not supported on this target";
LABEL_3:
    v7 = v4;
    v8 = 3;
    sub_38BE3D0(v2, a2, (__int64)&v7);
    return 0;
  }
  result = *(_QWORD *)(a1 + 72);
  if ( !result || *(_QWORD *)(result + 8) )
  {
    v9 = 1;
    v4 = ".seh_ directive must appear within an active frame";
    goto LABEL_3;
  }
  return result;
}
