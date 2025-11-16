// Function: sub_2246690
// Address: 0x2246690
//
__int64 __fastcall sub_2246690(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5, unsigned __int8 a6)
{
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r8
  wchar_t *v15; // r10
  wchar_t *v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rax
  size_t v19; // r14
  void *v20; // rsp
  char v21; // r8
  wchar_t *v22; // r11
  __int64 v23; // rax
  wchar_t *v24; // rsi
  __int64 v25; // rax
  wchar_t v26; // [rsp+0h] [rbp-A0h] BYREF
  int v27; // [rsp+4h] [rbp-9Ch]
  wchar_t *v28; // [rsp+8h] [rbp-98h]
  wchar_t *v29; // [rsp+10h] [rbp-90h]
  wchar_t *v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+50h] [rbp-50h]
  __int64 v32; // [rsp+58h] [rbp-48h]
  char v33[49]; // [rsp+6Fh] [rbp-31h] BYREF

  if ( (*(_DWORD *)(a4 + 24) & 1) == 0 )
  {
    result = sub_2246400(a1, a2, a3, a4, a5, a6);
    v31 = result;
    v32 = v12;
    return result;
  }
  LODWORD(v30) = *(_DWORD *)(a4 + 24);
  v13 = sub_22462F0((__int64)v33, (__int64 *)(a4 + 208));
  v14 = (unsigned int)v30;
  if ( a6 )
  {
    v15 = *(wchar_t **)(v13 + 40);
    v16 = (wchar_t *)*(int *)(v13 + 48);
  }
  else
  {
    v15 = *(wchar_t **)(v13 + 56);
    v16 = (wchar_t *)*(int *)(v13 + 64);
  }
  v17 = *(_QWORD *)(a4 + 16);
  if ( v17 <= (__int64)v16 )
  {
    *(_QWORD *)(a4 + 16) = 0;
    if ( !a3 )
    {
      v18 = *(_QWORD *)a2;
      LOBYTE(v29) = 0;
      (*(__int64 (__fastcall **)(__int64, wchar_t *, wchar_t *, wchar_t *, __int64))(v18 + 96))(a2, v15, v16, v16, v14);
    }
    return a2;
  }
  v19 = v17 - (_QWORD)v16;
  HIBYTE(v26) = a3;
  v27 = (int)v30;
  v28 = v15;
  v20 = alloca(4 * v19 + 8);
  v29 = v16;
  v30 = &v26;
  wmemset(&v26, a5, v19);
  v21 = v27;
  *(_QWORD *)(a4 + 16) = 0;
  v22 = v30;
  if ( (v21 & 0xB0) == 0x20 )
  {
    if ( HIBYTE(v26) )
      return a2;
    v25 = (*(__int64 (__fastcall **)(__int64, wchar_t *, wchar_t *))(*(_QWORD *)a2 + 96LL))(a2, v28, v29);
    if ( (wchar_t *)v25 != v29 )
      return a2;
    (*(void (__fastcall **)(__int64, wchar_t *, _QWORD))(*(_QWORD *)a2 + 96LL))(a2, v30, (int)v19);
    return a2;
  }
  else
  {
    v30 = v28;
    if ( HIBYTE(v26)
      || (int)v19 != (*(__int64 (__fastcall **)(__int64, wchar_t *, _QWORD, wchar_t *))(*(_QWORD *)a2 + 96LL))(
                       a2,
                       v22,
                       (int)v19,
                       v29) )
    {
      return a2;
    }
    v23 = *(_QWORD *)a2;
    v24 = v30;
    v30 = v29;
    (*(void (__fastcall **)(__int64, wchar_t *, wchar_t *))(v23 + 96))(a2, v24, v29);
    return a2;
  }
}
