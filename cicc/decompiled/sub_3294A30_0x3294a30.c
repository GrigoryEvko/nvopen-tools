// Function: sub_3294A30
// Address: 0x3294a30
//
__int64 __fastcall sub_3294A30(_QWORD *a1, _DWORD *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r9
  __int64 v7; // rsi
  int v8; // eax
  __int64 result; // rax
  __int64 (*v10)(); // rax
  __int64 (*v11)(); // rax
  __int64 v12; // rdx

  v6 = *(_QWORD *)(a5 + 16);
  v7 = *a1;
  v8 = *(_DWORD *)(*a1 + 24LL);
  if ( v8 != 214 )
  {
    if ( v8 != 213 )
      return 0;
    if ( *a2 )
      return 0;
    v10 = *(__int64 (**)())(*(_QWORD *)v6 + 704LL);
    if ( v10 == sub_2FE3250 )
      return 0;
    result = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, __int64))v10)(v6, v7, a1[1], a3, a4);
    if ( !(_BYTE)result )
      return 0;
LABEL_14:
    v12 = *(_QWORD *)(*a1 + 40LL);
    *a1 = *(_QWORD *)v12;
    *((_DWORD *)a1 + 2) = *(_DWORD *)(v12 + 8);
    return result;
  }
  v11 = *(__int64 (**)())(*(_QWORD *)v6 + 704LL);
  if ( v11 != sub_2FE3250 )
  {
    result = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, __int64))v11)(v6, v7, a1[1], a3, a4);
    if ( (_BYTE)result )
    {
      *a2 = 1;
      goto LABEL_14;
    }
  }
  result = 0;
  if ( !*a2 )
  {
    *a2 = 1;
    return 1;
  }
  return result;
}
