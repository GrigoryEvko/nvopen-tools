// Function: sub_1348DA0
// Address: 0x1348da0
//
__int64 __fastcall sub_1348DA0(__int64 *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  unsigned __int64 v5; // r11
  unsigned int v7; // r9d
  __int64 v8; // r10
  _QWORD *v9; // rdx
  __int64 v10; // r8
  __int64 *v11; // rcx
  unsigned __int64 v12; // r10
  __int64 v13; // r10

  a2[5] = 0;
  a2[6] = 0;
  a2[7] = 0;
  result = *a1;
  if ( *a1 )
  {
    if ( (a2[1] > *(_QWORD *)(result + 8)) - (a2[1] < *(_QWORD *)(result + 8)) == -1 )
    {
      a2[7] = result;
      *(_QWORD *)(result + 40) = a2;
      *a1 = (__int64)a2;
      a1[1] = 0;
      return result;
    }
    ++a1[1];
    a2[6] = *(_QWORD *)(result + 48);
    result = *a1;
    v3 = *(_QWORD *)(*a1 + 48);
    v4 = *a1 + 40;
    if ( v3 )
    {
      *(_QWORD *)(v3 + 40) = a2;
      result = *a1;
      v4 = *a1 + 40;
    }
    a2[5] = result;
    *(_QWORD *)(v4 + 8) = a2;
  }
  else
  {
    *a1 = (__int64)a2;
  }
  v5 = a1[1];
  if ( v5 > 1 )
  {
    result = -1;
    if ( !_BitScanForward64(&v5, v5 - 1) )
      LODWORD(v5) = -1;
    if ( (_DWORD)v5 )
    {
      v7 = 0;
      do
      {
        result = *a1;
        v9 = *(_QWORD **)(*a1 + 48);
        if ( !v9 )
          break;
        result = v9[6];
        if ( !result )
          break;
        v10 = *(_QWORD *)(result + 48);
        v9[6] = 0;
        v11 = (__int64 *)(result + 40);
        v9[5] = 0;
        *(_QWORD *)(result + 48) = 0;
        v12 = *(_QWORD *)(result + 8);
        *(_QWORD *)(result + 40) = 0;
        if ( (v9[1] > v12) - (v9[1] < v12) == -1 )
        {
          *(_QWORD *)(result + 40) = v9;
          v13 = v9[7];
          *(_QWORD *)(result + 48) = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 40) = result;
          v11 = v9 + 5;
          v9[7] = result;
          result = (__int64)v9;
          v9[6] = v10;
          if ( !v10 )
            goto LABEL_22;
        }
        else
        {
          v9[5] = result;
          v8 = *(_QWORD *)(result + 56);
          v9[6] = v8;
          if ( v8 )
            *(_QWORD *)(v8 + 40) = v9;
          *(_QWORD *)(result + 56) = v9;
          *(_QWORD *)(result + 48) = v10;
          if ( !v10 )
          {
LABEL_22:
            *(_QWORD *)(*a1 + 48) = result;
            result = *a1;
            *v11 = *a1;
            return result;
          }
        }
        *(_QWORD *)(v10 + 40) = result;
        ++v7;
        *(_QWORD *)(*a1 + 48) = result;
        result = *a1;
        *v11 = *a1;
      }
      while ( v7 < (unsigned int)v5 );
    }
  }
  return result;
}
