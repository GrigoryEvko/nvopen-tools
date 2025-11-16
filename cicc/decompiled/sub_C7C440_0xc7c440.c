// Function: sub_C7C440
// Address: 0xc7c440
//
__int64 __fastcall sub_C7C440(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v4; // ebx
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // esi
  _BYTE *v8; // rsi
  _BYTE *v9; // rdx
  bool v10; // cf
  _BYTE *v11; // rsi
  _BYTE *v12; // rdx

  result = *((unsigned int *)a1 + 2);
  if ( (_DWORD)result )
  {
    v4 = result - 1;
    while ( 1 )
    {
      v5 = *a1;
      v6 = 1LL << v4;
      if ( (unsigned int)result > 0x40 )
        v5 = *(_QWORD *)(v5 + 8LL * (v4 >> 6));
      v7 = *((_DWORD *)a1 + 6);
      result = a1[2];
      if ( (v5 & v6) != 0 )
        break;
      if ( v7 > 0x40 )
        result = *(_QWORD *)(result + 8LL * (v4 >> 6));
      v11 = *(_BYTE **)(a2 + 24);
      v12 = *(_BYTE **)(a2 + 32);
      if ( (result & v6) == 0 )
      {
        if ( v11 == v12 )
        {
          result = sub_CB6200(a2, "?", 1);
        }
        else
        {
          *v12 = 63;
          ++*(_QWORD *)(a2 + 32);
        }
        goto LABEL_11;
      }
      if ( v11 == v12 )
      {
        result = sub_CB6200(a2, "1", 1);
        goto LABEL_11;
      }
      *v12 = 49;
      ++*(_QWORD *)(a2 + 32);
      v10 = v4-- == 0;
      if ( v10 )
        return result;
LABEL_12:
      LODWORD(result) = *((_DWORD *)a1 + 2);
    }
    if ( v7 > 0x40 )
      result = *(_QWORD *)(result + 8LL * (v4 >> 6));
    v8 = *(_BYTE **)(a2 + 24);
    v9 = *(_BYTE **)(a2 + 32);
    if ( (result & v6) != 0 )
    {
      if ( v9 == v8 )
      {
        result = sub_CB6200(a2, &unk_3F6A4C5, 1);
      }
      else
      {
        *v9 = 33;
        ++*(_QWORD *)(a2 + 32);
      }
    }
    else if ( v9 == v8 )
    {
      result = sub_CB6200(a2, "0", 1);
    }
    else
    {
      *v9 = 48;
      ++*(_QWORD *)(a2 + 32);
    }
LABEL_11:
    v10 = v4-- == 0;
    if ( v10 )
      return result;
    goto LABEL_12;
  }
  return result;
}
