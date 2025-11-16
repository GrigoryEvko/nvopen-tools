// Function: sub_9B7C00
// Address: 0x9b7c00
//
__int64 __fastcall sub_9B7C00(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  char v4; // dl
  char v5; // al
  __int64 result; // rax
  __int64 v7; // r13
  _BYTE *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // ecx

LABEL_1:
  while ( 2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = *(_BYTE *)(v3 + 8);
        if ( v4 == 17 && (unsigned int)a2 >= *(_DWORD *)(v3 + 32) )
          return sub_ACADE0(*(_QWORD *)(v3 + 24));
        v5 = *(_BYTE *)a1;
        if ( *(_BYTE *)a1 <= 0x15u )
          return sub_AD69F0(a1, a2);
        if ( v5 != 91 )
          break;
        v10 = *(_QWORD *)(a1 - 32);
        if ( *(_BYTE *)v10 != 17 )
          return 0;
        if ( *(_DWORD *)(v10 + 32) <= 0x40u )
          v11 = *(_QWORD *)(v10 + 24);
        else
          v11 = **(_QWORD **)(v10 + 24);
        if ( (_DWORD)a2 == (_DWORD)v11 )
          return *(_QWORD *)(a1 - 64);
        v12 = *(_QWORD *)(a1 - 96);
        if ( v12 )
        {
          if ( a1 == v12 )
            return 0;
          v3 = *(_QWORD *)(v12 + 8);
          a1 = *(_QWORD *)(a1 - 96);
        }
        else
        {
          v3 = MEMORY[8];
          a1 = 0;
        }
      }
      if ( v5 != 92 )
        break;
      if ( v4 != 17 )
        goto LABEL_9;
      v13 = *(_QWORD *)(a1 - 64);
      a2 = *(unsigned int *)(*(_QWORD *)(a1 + 72) + 4LL * (unsigned int)a2);
      v14 = *(_DWORD *)(*(_QWORD *)(v13 + 8) + 32LL);
      if ( (int)a2 < 0 )
        return sub_ACADE0(*(_QWORD *)(v3 + 24));
      if ( v14 <= (int)a2 )
      {
        a1 = *(_QWORD *)(a1 - 32);
        a2 = (unsigned int)(a2 - v14);
        goto LABEL_1;
      }
      v3 = *(_QWORD *)(v13 + 8);
      a1 = *(_QWORD *)(a1 - 64);
    }
    if ( v5 == 42 )
    {
      v7 = *(_QWORD *)(a1 - 64);
      if ( v7 )
      {
        v8 = *(_BYTE **)(a1 - 32);
        if ( *v8 <= 0x15u )
        {
          v9 = sub_AD69F0(v8, a2);
          if ( v9 )
          {
            a2 = (unsigned int)a2;
            if ( (unsigned __int8)sub_AC30F0(v9) )
            {
              a1 = v7;
              continue;
            }
          }
          v4 = *(_BYTE *)(v3 + 8);
        }
      }
    }
    break;
  }
LABEL_9:
  if ( v4 != 18 )
    return 0;
  result = sub_9B7920((char *)a1);
  if ( !result || (unsigned int)a2 >= *(_DWORD *)(v3 + 32) )
    return 0;
  return result;
}
