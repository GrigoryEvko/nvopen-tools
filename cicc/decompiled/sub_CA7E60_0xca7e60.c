// Function: sub_CA7E60
// Address: 0xca7e60
//
__int64 __fastcall sub_CA7E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v5; // r12
  char *v6; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  const char *v10; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  v5 = *(char **)(a1 + 40);
  v6 = *(char **)(a1 + 48);
  if ( (unsigned int)a2 > 0x7F )
    goto LABEL_6;
  if ( v5 == v6 )
    return 0;
  if ( *v5 < 0 )
  {
LABEL_6:
    v8 = *(_QWORD *)(a1 + 336);
    v12 = 1;
    v10 = "Cannot consume non-ascii characters";
    v11 = 3;
    if ( v5 >= v6 )
      v5 = v6 - 1;
    if ( v8 )
    {
      v9 = sub_2241E50(a1, a2, v6, a4, a5);
      *(_DWORD *)v8 = 22;
      *(_QWORD *)(v8 + 8) = v9;
    }
    if ( !*(_BYTE *)(a1 + 75) )
      sub_C91CB0(*(__int64 **)a1, (unsigned __int64)v5, 0, (__int64)&v10, 0, 0, 0, 0, *(_BYTE *)(a1 + 76));
    *(_BYTE *)(a1 + 75) = 1;
    return 0;
  }
  else if ( *v5 == (_DWORD)a2 )
  {
    ++*(_DWORD *)(a1 + 60);
    *(_QWORD *)(a1 + 40) = v5 + 1;
    return 1;
  }
  else
  {
    return 0;
  }
}
