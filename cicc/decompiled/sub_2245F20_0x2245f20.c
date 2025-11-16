// Function: sub_2245F20
// Address: 0x2245f20
//
__int64 __fastcall sub_2245F20(__int64 a1, __int64 a2, wchar_t *a3, const wchar_t *a4, __int64 a5, signed __int64 a6)
{
  size_t v6; // r8
  const wchar_t *v7; // r14
  size_t v8; // r12
  wchar_t *v9; // rbp
  __int64 result; // rax
  __int64 v12; // r13
  __int64 v13; // rcx
  size_t v14; // rdx
  int v15; // eax
  wchar_t v16; // eax
  __int64 v17; // [rsp+0h] [rbp-48h]
  __int64 v18; // [rsp+0h] [rbp-48h]

  v6 = a5 - a6;
  v7 = a4;
  v8 = v6;
  v9 = a3;
  result = *(_DWORD *)(a1 + 24) & 0xB0;
  if ( (_DWORD)result == 32 )
  {
    if ( a6 )
    {
      result = (__int64)wmemcpy(a3, a4, a6);
      if ( !v8 )
        return result;
    }
    else if ( !v6 )
    {
      return result;
    }
    return (__int64)wmemset(&v9[a6], a2, v8);
  }
  v12 = 4 * v6;
  if ( (_DWORD)result != 16 )
    goto LABEL_3;
  v17 = sub_2243120((_QWORD *)(a1 + 208), a2);
  result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 80LL))(v17, 45);
  if ( *v7 == (_DWORD)result
    || (result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 80LL))(v17, 43), *v7 == (_DWORD)result) )
  {
    *v9 = result;
    ++v7;
    ++v9;
    v13 = 1;
    goto LABEL_4;
  }
  result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 80LL))(v17, 48);
  if ( *v7 != (_DWORD)result || a6 <= 1 )
  {
LABEL_3:
    v13 = 0;
    goto LABEL_4;
  }
  if ( v7[1] == (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 80LL))(v17, 120)
    || (v15 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v17 + 80LL))(v17, 88), v14 = a6, v7[1] == v15) )
  {
    v16 = *v7;
    v9 += 2;
    v7 += 2;
    v13 = 2;
    *(v9 - 2) = v16;
    result = *((unsigned int *)v7 - 1);
    *(v9 - 1) = result;
LABEL_4:
    if ( !v8 )
    {
      v14 = a6 - v13;
      if ( a6 == v13 )
        return result;
      goto LABEL_15;
    }
    goto LABEL_14;
  }
  v13 = 0;
  if ( !v8 )
    return (__int64)wmemcpy(v9, v7, v14);
LABEL_14:
  v18 = v13;
  result = (__int64)wmemset(v9, a2, v8);
  v14 = a6 - v18;
  if ( a6 != v18 )
  {
LABEL_15:
    v9 = (wchar_t *)((char *)v9 + v12);
    return (__int64)wmemcpy(v9, v7, v14);
  }
  return result;
}
