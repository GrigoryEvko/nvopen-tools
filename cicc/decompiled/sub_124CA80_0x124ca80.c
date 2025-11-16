// Function: sub_124CA80
// Address: 0x124ca80
//
__int64 __fastcall sub_124CA80(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbp
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  const char *v8; // rax
  const char *v10; // [rsp-38h] [rbp-38h] BYREF
  char v11; // [rsp-18h] [rbp-18h]
  char v12; // [rsp-17h] [rbp-17h]
  __int64 v13; // [rsp-8h] [rbp-8h]

  if ( !*(_QWORD *)(a1 + 128) )
    return 1;
  v13 = v5;
  v6 = *(_QWORD *)(a4 + 136);
  if ( v6 > 3 && *(_DWORD *)(*(_QWORD *)(a4 + 128) + v6 - 4) == 1870095406 )
  {
    v12 = 1;
    v8 = "A dwo section may not contain relocations";
    goto LABEL_9;
  }
  if ( a5 )
  {
    v7 = *(_QWORD *)(a5 + 136);
    if ( v7 > 3 && *(_DWORD *)(*(_QWORD *)(a5 + 128) + v7 - 4) == 1870095406 )
    {
      v12 = 1;
      v8 = "A relocation may not refer to a dwo section";
LABEL_9:
      v10 = v8;
      v11 = 3;
      sub_E66880(a2, a3, (__int64)&v10);
      return 0;
    }
  }
  return 1;
}
