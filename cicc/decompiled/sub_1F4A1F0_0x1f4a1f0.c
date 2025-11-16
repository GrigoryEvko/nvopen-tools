// Function: sub_1F4A1F0
// Address: 0x1f4a1f0
//
_BYTE *__fastcall sub_1F4A1F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 *v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  bool v9; // zf
  const char *v10; // rdi
  size_t v11; // rax
  _BYTE *result; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  char *v14[2]; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v15[3]; // [rsp+20h] [rbp-20h] BYREF

  v5 = *a1;
  v6 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 8) + 24LL) + 16LL * (*(_DWORD *)*a1 & 0x7FFFFFFF))
                 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 8) + 24LL) + 16LL * (*(_DWORD *)*a1 & 0x7FFFFFFF)) & 4) != 0 )
  {
    if ( v6 )
    {
      v10 = (const char *)v6[1];
      v11 = 0;
      v13[0] = v10;
      if ( !v10 )
      {
LABEL_5:
        v13[1] = v11;
        sub_16D2060(v14, v13, v5, (__int64)v6, a5);
        sub_16E7EE0(a2, v14[0], (size_t)v14[1]);
        result = v15;
        if ( (_QWORD *)v14[0] != v15 )
          return (_BYTE *)j_j___libc_free_0(v14[0], v15[0] + 1LL);
        return result;
      }
LABEL_4:
      v11 = strlen(v10);
      goto LABEL_5;
    }
  }
  else if ( v6 )
  {
    v7 = *(_QWORD *)(v5 + 16);
    v5 = *v6;
    v8 = *(unsigned int *)(*v6 + 16);
    v9 = *(_QWORD *)(v7 + 80) + v8 == 0;
    v10 = (const char *)(*(_QWORD *)(v7 + 80) + v8);
    v11 = 0;
    v13[0] = v10;
    if ( v9 )
      goto LABEL_5;
    goto LABEL_4;
  }
  result = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a2, "_", 1u);
  *result = 95;
  ++*(_QWORD *)(a2 + 24);
  return result;
}
