// Function: sub_37DD360
// Address: 0x37dd360
//
__int64 __fastcall sub_37DD360(
        __int64 *a1,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        unsigned __int8 (__fastcall *a5)(__int64 *, __int64 *))
{
  bool v8; // zf
  __int64 v9; // rax
  int v10; // edx
  __int64 result; // rax
  int v12; // edx
  int v13; // edx

  if ( a5(a2, a3) )
  {
    if ( !a5(a3, a4) )
    {
      v8 = a5(a2, a4) == 0;
      v9 = *a1;
      if ( !v8 )
      {
LABEL_4:
        *a1 = *a4;
        v10 = *((_DWORD *)a4 + 2);
        *a4 = v9;
        result = *((unsigned int *)a1 + 2);
        *((_DWORD *)a1 + 2) = v10;
        *((_DWORD *)a4 + 2) = result;
        return result;
      }
      goto LABEL_9;
    }
    v9 = *a1;
LABEL_7:
    *a1 = *a3;
    v12 = *((_DWORD *)a3 + 2);
    *a3 = v9;
    result = *((unsigned int *)a1 + 2);
    *((_DWORD *)a1 + 2) = v12;
    *((_DWORD *)a3 + 2) = result;
    return result;
  }
  if ( !a5(a2, a4) )
  {
    v8 = a5(a3, a4) == 0;
    v9 = *a1;
    if ( !v8 )
      goto LABEL_4;
    goto LABEL_7;
  }
  v9 = *a1;
LABEL_9:
  *a1 = *a2;
  v13 = *((_DWORD *)a2 + 2);
  *a2 = v9;
  result = *((unsigned int *)a1 + 2);
  *((_DWORD *)a1 + 2) = v13;
  *((_DWORD *)a2 + 2) = result;
  return result;
}
