// Function: sub_7A7280
// Address: 0x7a7280
//
unsigned __int64 __fastcall sub_7A7280(__int64 a1, int a2)
{
  unsigned __int64 result; // rax
  __int64 **v3; // rcx
  __int64 *v4; // r8
  __int64 v5; // rax
  __int64 *v6; // rax
  unsigned __int64 v7; // r9
  unsigned __int64 v8; // rbx
  __int64 v9; // r12
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r8

  result = *(_QWORD *)a1;
  if ( !a2 && *(_QWORD *)(result + 160) )
    return result;
  result = *(_QWORD *)(result + 168);
  v3 = *(__int64 ***)result;
  if ( !*(_QWORD *)result )
    return result;
  while ( 1 )
  {
    result = *((unsigned __int8 *)v3 + 96);
    if ( (result & 1) != 0 && ((result & 2) != 0) == a2 && (result & 0x40) != 0 )
    {
      v4 = v3[5];
      v5 = *v4;
      if ( !*v4 )
        break;
      if ( *((_BYTE *)v4 + 140) == 12 )
      {
        v6 = v3[5];
        do
          v6 = (__int64 *)v6[20];
        while ( *((_BYTE *)v6 + 140) == 12 );
        v5 = *v6;
      }
      result = *(_QWORD *)(v5 + 96);
      if ( *(char *)(result + 178) >= 0 )
        break;
    }
    v3 = (__int64 **)*v3;
    if ( !v3 )
      return result;
  }
LABEL_15:
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(unsigned int *)(a1 + 24);
  v9 = *(_QWORD *)(v4[21] + 32);
  v10 = v7 % v8;
  result = v7;
  if ( !(v7 % v8) || v8 - v10 > unk_4F06AC0 )
  {
    if ( (__int64 *)((char *)v3[13] + v9) != (__int64 *)v7 )
      goto LABEL_18;
LABEL_34:
    result = v4[20];
    if ( result )
    {
      v13 = 0;
      do
      {
        if ( (*(_BYTE *)(result + 144) & 0x40) == 0 )
          v13 = result;
        result = *(_QWORD *)(result + 112);
      }
      while ( result );
      if ( v13 )
      {
        if ( (*(_BYTE *)(v13 + 144) & 4) != 0 )
        {
          v14 = *(unsigned __int8 *)(v13 + 137);
          if ( (_BYTE)v14 )
          {
            result = dword_4F06BA0 * (v9 - *(_QWORD *)(v13 + 128)) - (*(unsigned __int8 *)(v13 + 136) + v14);
            if ( dword_4F06BA0 > result )
            {
              *(_QWORD *)(a1 + 8) = v8 + v7;
              return result;
            }
          }
        }
      }
    }
    goto LABEL_18;
  }
  result = v7 + v8 - v10;
  if ( v7 > v10 - v8 + unk_4F06AC0 )
    result = *(_QWORD *)(a1 + 8);
  if ( (__int64 *)((char *)v3[13] + v9) == (__int64 *)result )
    goto LABEL_34;
LABEL_18:
  while ( 1 )
  {
    v3 = (__int64 **)*v3;
    if ( !v3 )
      return result;
    while ( 1 )
    {
      result = *((unsigned __int8 *)v3 + 96);
      if ( (result & 1) == 0 || ((result & 2) != 0) != a2 || (result & 0x40) == 0 )
        break;
      v4 = v3[5];
      v11 = *v4;
      if ( !*v4 )
        goto LABEL_15;
      if ( *((_BYTE *)v4 + 140) == 12 )
      {
        v12 = v3[5];
        do
          v12 = (__int64 *)v12[20];
        while ( *((_BYTE *)v12 + 140) == 12 );
        v11 = *v12;
      }
      result = *(_QWORD *)(v11 + 96);
      if ( *(char *)(result + 178) >= 0 )
        goto LABEL_15;
      v3 = (__int64 **)*v3;
      if ( !v3 )
        return result;
    }
  }
}
