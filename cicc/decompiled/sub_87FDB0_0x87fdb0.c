// Function: sub_87FDB0
// Address: 0x87fdb0
//
_QWORD *__fastcall sub_87FDB0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  char v4; // si
  char v5; // cl
  _QWORD *result; // rax
  __int64 v7; // r14
  __int64 *v8; // rdx
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  int v13; // edx

  for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v2 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v3 = *(_QWORD *)(v2 + 144);
  if ( v3 )
  {
    while ( *(_QWORD *)v3 != *a1 )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        goto LABEL_22;
    }
    v4 = *(_BYTE *)(v3 + 80);
    if ( v4 != 17 )
    {
      v5 = *(_BYTE *)(v3 + 80);
      result = (_QWORD *)v3;
      v7 = 0;
      goto LABEL_12;
    }
    result = *(_QWORD **)(v3 + 88);
    if ( result )
    {
      v5 = *((_BYTE *)result + 80);
      v7 = v3;
      while ( 1 )
      {
LABEL_12:
        v8 = result;
        if ( v5 == 16 )
        {
          v8 = *(__int64 **)result[11];
          v5 = *((_BYTE *)v8 + 80);
        }
        if ( v5 == 24 )
          v8 = (__int64 *)v8[11];
        if ( a1 == v8 )
          break;
        if ( v4 != 17 || (result = (_QWORD *)result[1]) == 0 )
        {
          v9 = sub_87ECE0(a1, a1 + 6, dword_4F04C34);
          if ( v7 )
            goto LABEL_19;
          v10 = *(_QWORD *)(v2 + 144);
          v11 = *(_QWORD *)(v3 + 8);
          if ( v10 == v3 )
          {
            *(_QWORD *)(v2 + 144) = v11;
          }
          else
          {
            do
            {
              v12 = v10;
              v10 = *(_QWORD *)(v10 + 8);
            }
            while ( v10 != v3 );
            *(_QWORD *)(v12 + 8) = v11;
          }
          *(_QWORD *)(v3 + 8) = 0;
          result = sub_87EBB0(0x11u, *v9, (_QWORD *)(v3 + 48));
          v13 = *((_DWORD *)v9 + 10);
          result[11] = v9;
          *((_DWORD *)result + 10) = v13;
          v9[1] = v3;
          *((_BYTE *)v9 + 83) |= 0x20u;
          *(_BYTE *)(v3 + 83) |= 0x20u;
          result[1] = *(_QWORD *)(v2 + 144);
          *(_QWORD *)(v2 + 144) = result;
          return result;
        }
        v5 = *((_BYTE *)result + 80);
      }
    }
    else
    {
      v7 = v3;
      v9 = sub_87ECE0(a1, a1 + 6, dword_4F04C34);
LABEL_19:
      result = *(_QWORD **)(v7 + 88);
      v9[1] = (__int64)result;
      *(_QWORD *)(v7 + 88) = v9;
      *((_BYTE *)v9 + 83) |= 0x20u;
    }
  }
  else
  {
LABEL_22:
    result = sub_87ECE0(a1, a1 + 6, dword_4F04C34);
    result[1] = *(_QWORD *)(v2 + 144);
    *(_QWORD *)(v2 + 144) = result;
  }
  return result;
}
