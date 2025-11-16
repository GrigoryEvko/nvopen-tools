// Function: sub_C7ECD0
// Address: 0xc7ecd0
//
__int64 __fastcall sub_C7ECD0(__int64 a1, char a2)
{
  __int64 v2; // r12
  char v3; // bl
  __int64 v5; // rdx
  _DWORD *v6; // rdx
  void *v7; // rdx
  _WORD *v8; // rdx
  __int64 v9; // rax
  void *v10; // rdx
  void *v11; // rdx
  _WORD *v12; // rdx
  __int64 v13; // rax

  v2 = a1;
  if ( a2 )
  {
    v3 = a2 & 0xC;
    if ( (a2 & 3) == 1 )
    {
      v10 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v10 <= 0xEu )
      {
        sub_CB6200(a1, "address_is_null", 15);
      }
      else
      {
        qmemcpy(v10, "address_is_null", 15);
        *(_QWORD *)(a1 + 32) += 15LL;
      }
    }
    else
    {
      if ( (a2 & 3) == 0 )
      {
        if ( v3 != 4 )
        {
          if ( v3 != 12 )
            return v2;
          v7 = *(void **)(a1 + 32);
          goto LABEL_12;
        }
        v11 = *(void **)(a1 + 32);
LABEL_23:
        if ( *(_QWORD *)(a1 + 24) - (_QWORD)v11 <= 0xEu )
        {
          sub_CB6200(a1, "read_provenance", 15);
        }
        else
        {
          qmemcpy(v11, "read_provenance", 15);
          *(_QWORD *)(a1 + 32) += 15LL;
        }
        return v2;
      }
      v5 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v5) <= 6 )
      {
        sub_CB6200(a1, "address", 7);
      }
      else
      {
        *(_DWORD *)v5 = 1919181921;
        *(_WORD *)(v5 + 4) = 29541;
        *(_BYTE *)(v5 + 6) = 115;
        *(_QWORD *)(a1 + 32) += 7LL;
      }
    }
    if ( v3 != 4 )
    {
      if ( v3 != 12 )
        return v2;
      v8 = *(_WORD **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v8 > 1u )
      {
        *v8 = 8236;
        v7 = (void *)(*(_QWORD *)(a1 + 32) + 2LL);
        *(_QWORD *)(a1 + 32) = v7;
      }
      else
      {
        v9 = sub_CB6200(a1, ", ", 2);
        v7 = *(void **)(v9 + 32);
        a1 = v9;
      }
LABEL_12:
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v7 <= 9u )
      {
        sub_CB6200(a1, "provenance", 10);
      }
      else
      {
        qmemcpy(v7, "provenance", 10);
        *(_QWORD *)(a1 + 32) += 10LL;
      }
      return v2;
    }
    v12 = *(_WORD **)(a1 + 32);
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v12 > 1u )
    {
      *v12 = 8236;
      v11 = (void *)(*(_QWORD *)(a1 + 32) + 2LL);
      *(_QWORD *)(a1 + 32) = v11;
    }
    else
    {
      v13 = sub_CB6200(a1, ", ", 2);
      v11 = *(void **)(v13 + 32);
      a1 = v13;
    }
    goto LABEL_23;
  }
  v6 = *(_DWORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v6 <= 3u )
  {
    sub_CB6200(a1, "none", 4);
  }
  else
  {
    *v6 = 1701736302;
    *(_QWORD *)(a1 + 32) += 4LL;
  }
  return v2;
}
