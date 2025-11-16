// Function: sub_3170360
// Address: 0x3170360
//
__int64 __fastcall sub_3170360(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rbx

  if ( *(_BYTE *)(a1 + 344) )
  {
    v6 = *(_DWORD *)(a1 + 360);
    if ( v6 <= 0x40 )
    {
      if ( !*(_QWORD *)(a1 + 352) )
        goto LABEL_4;
    }
    else if ( v6 == (unsigned int)sub_C444A0(a1 + 352) )
    {
LABEL_4:
      v7 = *(_QWORD *)(a2 - 32);
      if ( !v7 || *(_BYTE *)v7 )
        BUG();
      v8 = *(_QWORD *)(a2 + 80);
      if ( *(_QWORD *)(v7 + 24) == v8 )
      {
        result = *(unsigned int *)(v7 + 36);
        if ( (_DWORD)result != 210 )
        {
          if ( (_DWORD)result != 211 )
          {
            if ( (_DWORD)result != 171 )
              return sub_31700B0(a1, (unsigned __int8 *)a2);
            goto LABEL_19;
          }
          if ( !*(_BYTE *)(a1 + 516) )
            goto LABEL_37;
          v12 = *(__int64 **)(a1 + 496);
          v8 = *(unsigned int *)(a1 + 508);
          a3 = &v12[v8];
          if ( v12 != a3 )
          {
            while ( a2 != *v12 )
            {
              if ( a3 == ++v12 )
                goto LABEL_36;
            }
            goto LABEL_33;
          }
LABEL_36:
          if ( (unsigned int)v8 < *(_DWORD *)(a1 + 504) )
          {
            *(_DWORD *)(a1 + 508) = v8 + 1;
            *a3 = a2;
            ++*(_QWORD *)(a1 + 488);
          }
          else
          {
LABEL_37:
            sub_C8CC70(a1 + 488, a2, (__int64)a3, v8, a5, a6);
          }
LABEL_33:
          result = *(unsigned int *)(a1 + 544);
          v13 = *(_QWORD *)(a2 + 40);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 548) )
          {
            sub_C8D5F0(a1 + 536, (const void *)(a1 + 552), result + 1, 8u, a5, a6);
            result = *(unsigned int *)(a1 + 544);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 536) + 8 * result) = v13;
          ++*(_DWORD *)(a1 + 544);
          return result;
        }
        v11 = *(_QWORD *)(a2 + 40);
        if ( *(_BYTE *)(a1 + 628) )
        {
          result = *(_QWORD *)(a1 + 608);
          v8 = *(unsigned int *)(a1 + 620);
          a3 = (__int64 *)(result + 8 * v8);
          if ( (__int64 *)result != a3 )
          {
            while ( v11 != *(_QWORD *)result )
            {
              result += 8;
              if ( a3 == (__int64 *)result )
                goto LABEL_25;
            }
            return result;
          }
LABEL_25:
          if ( (unsigned int)v8 < *(_DWORD *)(a1 + 616) )
          {
            *(_DWORD *)(a1 + 620) = v8 + 1;
            *a3 = v11;
            ++*(_QWORD *)(a1 + 600);
            return result;
          }
        }
        return (__int64)sub_C8CC70(a1 + 600, v11, (__int64)a3, v8, a5, a6);
      }
LABEL_39:
      BUG();
    }
  }
  v10 = *(_QWORD *)(a2 - 32);
  if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_39;
  result = *(unsigned int *)(v10 + 36);
  if ( (_DWORD)result == 171 )
  {
LABEL_19:
    *(_QWORD *)(a1 + 16) = a2;
    return result;
  }
  result = (unsigned int)(result - 210);
  if ( (unsigned int)result > 1 )
    return sub_31700B0(a1, (unsigned __int8 *)a2);
  return result;
}
