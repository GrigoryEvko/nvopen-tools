// Function: sub_C53C20
// Address: 0xc53c20
//
__int64 __fastcall sub_C53C20(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // rdx
  __int64 result; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdx

  if ( !*(_BYTE *)(a1 + 308) )
  {
LABEL_26:
    sub_C8CC70(a1 + 280, a2);
    goto LABEL_6;
  }
  v3 = *(__int64 **)(a1 + 288);
  v4 = *(unsigned int *)(a1 + 300);
  v5 = &v3[v4];
  if ( v3 == v5 )
  {
LABEL_25:
    if ( (unsigned int)v4 < *(_DWORD *)(a1 + 296) )
    {
      *(_DWORD *)(a1 + 300) = v4 + 1;
      *v5 = a2;
      ++*(_QWORD *)(a1 + 280);
      goto LABEL_6;
    }
    goto LABEL_26;
  }
  while ( a2 != *v3 )
  {
    if ( v5 == ++v3 )
      goto LABEL_25;
  }
LABEL_6:
  result = sub_C52D90();
  LODWORD(v9) = *(_DWORD *)(result + 136);
  if ( (_DWORD)v9 )
  {
    v10 = *(__int64 **)(result + 128);
    result = *v10;
    if ( *v10 != -8 && result )
    {
      v12 = (__int64)v10;
    }
    else
    {
      result = (__int64)(v10 + 1);
      do
      {
        do
        {
          v11 = *(_QWORD *)result;
          v12 = result;
          result += 8;
        }
        while ( !v11 );
      }
      while ( v11 == -8 );
    }
    v9 = (unsigned int)v9;
    v13 = &v10[(unsigned int)v9];
    if ( v13 != (__int64 *)v12 )
    {
      while ( 1 )
      {
        v14 = *(_QWORD *)(*(_QWORD *)v12 + 8LL);
        if ( ((*(_WORD *)(v14 + 12) >> 7) & 3) == 1
          || (*(_BYTE *)(v14 + 13) & 8) != 0
          || (*(_BYTE *)(v14 + 12) & 7) == 4
          || *(_QWORD *)(v14 + 32) )
        {
          sub_C538D0((_QWORD *)a1, v14, a2, v9, v7, v8);
        }
        else
        {
          sub_C52060((_QWORD *)a1, v14, a2, (const void *)(*(_QWORD *)v12 + 16LL), **(_QWORD **)v12);
        }
        v15 = *(_QWORD *)(v12 + 8);
        result = v12 + 8;
        if ( v15 == -8 || !v15 )
        {
          do
          {
            do
            {
              v16 = *(_QWORD *)(result + 8);
              result += 8;
            }
            while ( !v16 );
          }
          while ( v16 == -8 );
        }
        if ( (__int64 *)result == v13 )
          break;
        v12 = result;
      }
    }
  }
  return result;
}
