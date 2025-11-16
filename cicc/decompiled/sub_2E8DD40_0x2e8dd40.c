// Function: sub_2E8DD40
// Address: 0x2e8dd40
//
__int64 __fastcall sub_2E8DD40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi

  result = *(_QWORD *)(a1 + 32);
  if ( !*(_BYTE *)result )
  {
    v3 = a1;
    if ( (*(_BYTE *)a1 & 4) == 0 && (*(_BYTE *)(a1 + 44) & 8) != 0 )
    {
      do
        v3 = *(_QWORD *)(v3 + 8);
      while ( (*(_BYTE *)(v3 + 44) & 8) != 0 );
    }
    v6 = *(_QWORD *)(v3 + 8);
    result = *(_QWORD *)(a1 + 24);
    v7 = result + 48;
    if ( result + 48 != v6 )
    {
      result = *(unsigned __int16 *)(v6 + 68);
      if ( (_WORD)result == 14 )
        goto LABEL_13;
      while ( 1 )
      {
        if ( (_WORD)result != 15 )
          return result;
        v8 = *(_QWORD *)(v6 + 32);
        v9 = v8 + 40LL * (*(_DWORD *)(v6 + 40) & 0xFFFFFF);
        result = sub_2E85500(v8 + 80, v9, *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL));
        if ( v9 != result )
        {
LABEL_14:
          result = *(unsigned int *)(a2 + 8);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 8u, v10, v11);
            result = *(unsigned int *)(a2 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v6;
          ++*(_DWORD *)(a2 + 8);
        }
        while ( 1 )
        {
          if ( (*(_BYTE *)v6 & 4) != 0 )
          {
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              return result;
          }
          else
          {
            while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
              v6 = *(_QWORD *)(v6 + 8);
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              return result;
          }
          result = *(unsigned __int16 *)(v6 + 68);
          if ( (_WORD)result != 14 )
            break;
LABEL_13:
          v12 = *(_QWORD *)(v6 + 32);
          result = sub_2E85500(v12, v12 + 40, *(_DWORD *)(*(_QWORD *)(a1 + 32) + 8LL));
          if ( v12 + 40 != result )
            goto LABEL_14;
        }
      }
    }
  }
  return result;
}
