// Function: sub_1705300
// Address: 0x1705300
//
__int64 __fastcall sub_1705300(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  unsigned __int8 *v7; // rcx
  unsigned __int8 *v8; // rdx
  int v9; // esi
  int v10; // edx
  __int64 v11; // rdx
  char v12; // cl

  result = a1;
  if ( a2 != 1 )
  {
    v6 = *(_QWORD *)(result + 2664);
    result = *(_QWORD *)(v6 + 24);
    v7 = (unsigned __int8 *)(result + *(unsigned int *)(v6 + 32));
    if ( (unsigned __int8 *)result != v7 )
    {
      v8 = *(unsigned __int8 **)(v6 + 24);
      do
      {
        if ( a2 == *v8 )
        {
          v9 = 1;
          goto LABEL_7;
        }
        ++v8;
      }
      while ( v7 != v8 );
      v9 = 0;
LABEL_7:
      if ( a3 != 1 )
        goto LABEL_10;
      goto LABEL_15;
    }
    if ( a3 != 1 )
    {
      v10 = 0;
      v9 = 0;
      if ( a2 <= a3 )
        goto LABEL_16;
      goto LABEL_12;
    }
    v9 = 0;
LABEL_15:
    v10 = 1;
LABEL_16:
    LOBYTE(result) = a2 >= a3;
    return v10 | v9 | (unsigned int)result;
  }
  if ( a3 == 1 )
  {
    v9 = 1;
    goto LABEL_15;
  }
  v11 = *(_QWORD *)(result + 2664);
  v9 = 1;
  result = *(_QWORD *)(v11 + 24);
  v7 = (unsigned __int8 *)(result + *(unsigned int *)(v11 + 32));
  if ( (unsigned __int8 *)result == v7 )
  {
LABEL_19:
    if ( a2 <= a3 )
    {
      v12 = v9;
      v10 = 0;
      goto LABEL_21;
    }
    v10 = 0;
  }
  else
  {
LABEL_10:
    while ( *(unsigned __int8 *)result != a3 )
    {
      if ( v7 == (unsigned __int8 *)++result )
        goto LABEL_19;
    }
    v10 = 1;
    if ( a2 <= a3 )
      return 1;
  }
LABEL_12:
  result = (a3 - 8) & 0xFFFFFFF7;
  LOBYTE(result) = a3 == 32 || ((a3 - 8) & 0xFFFFFFF7) == 0;
  if ( (_BYTE)result )
    return result;
  v12 = v9 & (v10 ^ 1);
LABEL_21:
  result = 0;
  if ( !v12 )
    goto LABEL_16;
  return result;
}
