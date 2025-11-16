// Function: sub_31DF300
// Address: 0x31df300
//
__int64 __fastcall sub_31DF300(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 (__fastcall *v3)(__int64, __int64); // rdx
  char v4; // al

  if ( ((*(_BYTE *)(*(_QWORD *)(a1[29] + 8LL) + 879LL) & 0x10) != 0 || *(_BYTE *)(a2 + 260)) && !sub_2E31AB0(a2) )
    return 1;
  result = *(unsigned int *)(a2 + 72);
  if ( !(_DWORD)result )
    return result;
  v3 = *(__int64 (__fastcall **)(__int64, __int64))(*a1 + 376LL);
  if ( v3 == sub_31D6920 )
  {
    if ( *(_BYTE *)(a2 + 216) || (_DWORD)result != 1 )
      return 1;
    v4 = sub_31D6700(a2);
  }
  else
  {
    v4 = v3((__int64)a1, a2);
  }
  if ( !v4 || *(_BYTE *)(a2 + 235) )
    return 1;
  return *(unsigned __int8 *)(a2 + 232);
}
