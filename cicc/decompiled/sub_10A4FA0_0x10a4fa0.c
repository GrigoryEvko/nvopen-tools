// Function: sub_10A4FA0
// Address: 0x10a4fa0
//
__int64 __fastcall sub_10A4FA0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 *v4; // rax
  __int64 *v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 57 )
    return 0;
  v4 = *a1;
  v5 = *(__int64 **)(a2 - 32);
  if ( *(__int64 **)(a2 - 64) == *a1 )
  {
    v9 = v5[2];
    if ( v9 )
    {
      if ( !*(_QWORD *)(v9 + 8) && *(_BYTE *)v5 == 44 )
      {
        result = sub_10081F0(a1 + 1, *(v5 - 8));
        if ( (_BYTE)result )
        {
          v8 = *(v5 - 4);
          if ( v8 )
            goto LABEL_14;
        }
        v5 = *(__int64 **)(a2 - 32);
        v4 = *a1;
      }
    }
  }
  if ( v5 == v4 )
  {
    v6 = *(_QWORD *)(a2 - 64);
    v7 = *(_QWORD *)(v6 + 16);
    if ( v7 )
    {
      if ( !*(_QWORD *)(v7 + 8) && *(_BYTE *)v6 == 44 )
      {
        result = sub_10081F0(a1 + 1, *(_QWORD *)(v6 - 64));
        if ( (_BYTE)result )
        {
          v8 = *(_QWORD *)(v6 - 32);
          if ( v8 )
          {
LABEL_14:
            *a1[2] = v8;
            return result;
          }
        }
      }
    }
  }
  return 0;
}
