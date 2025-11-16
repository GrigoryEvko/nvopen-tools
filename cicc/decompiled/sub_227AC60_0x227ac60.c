// Function: sub_227AC60
// Address: 0x227ac60
//
_QWORD *__fastcall sub_227AC60(__int64 a1, __int64 a2)
{
  __int64 *v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 *v6; // rax
  __int64 v7; // rcx
  int v8; // eax
  _QWORD *result; // rax
  __int64 *v10; // rax
  int v11; // eax

  if ( !*(_BYTE *)(a1 + 76) )
  {
    v10 = sub_C8CA60(a1 + 48, a2);
    if ( !v10 )
    {
      v5 = *(unsigned int *)(a1 + 68);
      goto LABEL_10;
    }
    *v10 = -2;
    v11 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 48);
    v5 = *(unsigned int *)(a1 + 68);
    v8 = v11 + 1;
    *(_DWORD *)(a1 + 72) = v8;
LABEL_7:
    if ( v8 == (_DWORD)v5 )
      goto LABEL_11;
    return sub_AE6EC0(a1, a2);
  }
  v3 = *(__int64 **)(a1 + 56);
  v4 = (__int64)&v3[*(unsigned int *)(a1 + 68)];
  v5 = *(unsigned int *)(a1 + 68);
  v6 = v3;
  if ( v3 != (__int64 *)v4 )
  {
    while ( a2 != *v6 )
    {
      if ( (__int64 *)v4 == ++v6 )
        goto LABEL_10;
    }
    v7 = (unsigned int)(v5 - 1);
    *(_DWORD *)(a1 + 68) = v7;
    v4 = v3[v7];
    *v6 = v4;
    v5 = *(unsigned int *)(a1 + 68);
    ++*(_QWORD *)(a1 + 48);
    v8 = *(_DWORD *)(a1 + 72);
    goto LABEL_7;
  }
LABEL_10:
  if ( *(_DWORD *)(a1 + 72) != (_DWORD)v5 )
    return sub_AE6EC0(a1, a2);
LABEL_11:
  result = (_QWORD *)sub_B19060(a1, (__int64)&unk_4F82400, v4, v5);
  if ( !(_BYTE)result )
    return sub_AE6EC0(a1, a2);
  return result;
}
