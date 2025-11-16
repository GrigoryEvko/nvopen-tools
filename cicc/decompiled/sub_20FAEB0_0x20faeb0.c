// Function: sub_20FAEB0
// Address: 0x20faeb0
//
_QWORD *__fastcall sub_20FAEB0(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7[5]; // [rsp-28h] [rbp-28h] BYREF

  result = *(_QWORD **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  if ( result )
  {
    v3 = sub_15B1030((__int64)result);
    if ( *(_DWORD *)(a2 + 8) == 2 && (v4 = *(_QWORD *)(a2 - 8)) != 0 )
    {
      v7[0] = v3;
      v7[1] = v4;
      v5 = sub_20FAE40(a1 + 8, (unsigned __int64)(31 * v3 + v4) % a1[9], v7, 31 * v3 + v4);
      if ( v5 && (v6 = *v5) != 0 )
        return (_QWORD *)(v6 + 24);
      else
        return 0;
    }
    else
    {
      v7[0] = v3;
      result = sub_20FABF0(a1 + 1, v7);
      if ( result )
        result += 2;
    }
  }
  return result;
}
