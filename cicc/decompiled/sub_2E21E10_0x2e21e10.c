// Function: sub_2E21E10
// Address: 0x2e21e10
//
__int64 __fastcall sub_2E21E10(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r10
  __int64 v5; // rdi
  char v6; // r8
  unsigned __int16 v7; // cx
  unsigned __int16 v8; // dx

  result = *a1;
  v3 = *(unsigned int *)(*a1 + 44);
  if ( (_DWORD)v3 )
  {
    v5 = 0;
LABEL_3:
    v6 = v5;
    result = *(_QWORD *)(result + 48) + 4 * v5;
    v7 = *(_WORD *)result;
    v8 = *(_WORD *)(result + 2);
    do
    {
      if ( !v7 )
      {
        if ( v3 == ++v5 )
          return result;
LABEL_7:
        result = *a1;
        goto LABEL_3;
      }
      result = (unsigned int)(*(_DWORD *)(a2 + 4 * ((unsigned __int64)v7 >> 5)) >> v7);
      v7 = v8;
      v8 = 0;
    }
    while ( (result & 1) != 0 );
    result = (unsigned int)v5++ >> 6;
    *(_QWORD *)(a1[1] + 8 * result) &= ~(1LL << v6);
    if ( v3 != v5 )
      goto LABEL_7;
  }
  return result;
}
