// Function: sub_86CBE0
// Address: 0x86cbe0
//
unsigned int *__fastcall sub_86CBE0(__int64 a1)
{
  __int64 v2; // rdi
  char v3; // al
  __int64 v4; // rbx
  char v5; // dl
  char v6; // al
  unsigned int *result; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  char v11; // dl
  _QWORD *v12; // rcx
  __int64 v13; // rsi
  char v14; // al
  char v15; // al
  char v16; // al
  char v17; // al
  char v18; // al
  _QWORD *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 i; // r15
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  unsigned __int8 v29; // [rsp+7h] [rbp-39h] BYREF
  __int64 v30[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( qword_4F5FD70 )
  {
    v2 = qword_4F5FD68;
    v3 = *(_BYTE *)(a1 + 32);
    v4 = *(_QWORD *)(qword_4F5FD68 + 16);
    v5 = *(_BYTE *)(qword_4F5FD68 + 32);
    if ( v3 != 5 )
    {
      if ( !v5 )
        v4 = qword_4F5FD68;
      *(_QWORD *)(a1 + 16) = v4;
      switch ( v3 )
      {
        case 0:
          v15 = *(_BYTE *)(a1 + 72);
          if ( (v15 & 4) == 0 && (*(_BYTE *)(v4 + 72) & 0xC) != 0 )
          {
            v16 = v15 | 8;
            *(_BYTE *)(a1 + 72) = v16;
            *(_BYTE *)(a1 + 72) = *(_BYTE *)(v4 + 72) & 0x10 | v16 & 0xEF;
          }
          if ( (*(_BYTE *)(v4 + 73) & 1) != 0 )
            *(_BYTE *)(a1 + 73) |= 1u;
          goto LABEL_8;
        case 1:
          v14 = *(_BYTE *)(a1 + 56) & 1;
          if ( *(_QWORD *)(v4 + 16) || *(_QWORD *)(v4 + 64) )
          {
            if ( !v14 )
            {
LABEL_44:
              *(_BYTE *)(v4 + 72) |= 0x10u;
              goto LABEL_8;
            }
          }
          else if ( !v14 )
          {
LABEL_64:
            result = (unsigned int *)qword_4F5FD60;
            *(_QWORD *)a1 = qword_4F5FD60;
            qword_4F5FD60 = a1;
            return result;
          }
          *(_BYTE *)(v4 + 72) |= 2u;
          goto LABEL_44;
        case 2:
          do
          {
            ++*(_QWORD *)(v4 + 64);
            v8 = *(_QWORD *)(v4 + 16);
            if ( !v8 )
              break;
            ++*(_QWORD *)(v8 + 64);
            v4 = *(_QWORD *)(v8 + 16);
          }
          while ( v4 );
          goto LABEL_8;
        case 3:
          do
          {
            v6 = *(_BYTE *)(v4 + 72);
            if ( (v6 & 1) != 0 )
              break;
            *(_BYTE *)(v4 + 72) = v6 | 1;
            v4 = *(_QWORD *)(v4 + 16);
          }
          while ( v4 );
          goto LABEL_8;
        case 4:
          if ( (unsigned int)sub_86C240(v4, 0) )
            goto LABEL_64;
          v17 = *(_BYTE *)(v4 + 72);
          if ( (v17 & 0x10) == 0 )
            goto LABEL_64;
          *(_QWORD *)(v4 + 48) = a1;
          v2 = qword_4F5FD68;
          *(_BYTE *)(v4 + 72) = v17 & 0xEF;
          if ( (v17 & 4) == 0 )
          {
            do
            {
              v4 = *(_QWORD *)(v4 + 16);
              v18 = *(_BYTE *)(v4 + 72);
              *(_QWORD *)(v4 + 48) = a1;
              *(_BYTE *)(v4 + 72) = v18 & 0xEF;
            }
            while ( (v18 & 4) == 0 );
          }
          goto LABEL_8;
        default:
          sub_721090();
      }
    }
    if ( !v5 )
    {
      if ( dword_4F077C4 == 2 )
      {
        sub_86C100(qword_4F5FD68);
        v2 = qword_4F5FD68;
      }
      v19 = *(_QWORD **)(v2 + 8);
      if ( v19 )
      {
        *v19 = *(_QWORD *)v2;
        v19 = *(_QWORD **)(v2 + 8);
      }
      else
      {
        qword_4F5FD70 = *(_QWORD *)v2;
      }
      if ( *(_QWORD *)v2 )
        *(_QWORD *)(*(_QWORD *)v2 + 8LL) = v19;
      else
        qword_4F5FD68 = (__int64)v19;
      result = (unsigned int *)qword_4F5FD60;
      *(_QWORD *)v2 = qword_4F5FD60;
      *(_QWORD *)a1 = v2;
      qword_4F5FD60 = a1;
      return result;
    }
    if ( dword_4F077C4 == 2 )
      sub_86C100(*(_QWORD *)(qword_4F5FD68 + 16));
    if ( (*(_BYTE *)(v4 + 72) & 1) == 0 && !*(_QWORD *)(v4 + 48) && !*(_QWORD *)(v4 + 64) )
    {
      v26 = qword_4F5FD68;
      v27 = *(_QWORD **)(v4 + 8);
      if ( v27 )
      {
        *v27 = *(_QWORD *)qword_4F5FD68;
        v27 = *(_QWORD **)(v4 + 8);
      }
      else
      {
        qword_4F5FD70 = *(_QWORD *)qword_4F5FD68;
      }
      if ( *(_QWORD *)v26 )
        *(_QWORD *)(*(_QWORD *)v26 + 8LL) = v27;
      else
        qword_4F5FD68 = (__int64)v27;
      result = (unsigned int *)qword_4F5FD60;
      *(_QWORD *)v26 = qword_4F5FD60;
      *(_QWORD *)a1 = v4;
      qword_4F5FD60 = a1;
      return result;
    }
    *(_BYTE *)(v4 + 72) &= ~0x10u;
    *(_QWORD *)(a1 + 40) = v4;
    v9 = *(_QWORD *)(v4 + 16);
    *(_QWORD *)(v4 + 40) = a1;
    v2 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 16) = v9;
    if ( (*(_BYTE *)(v2 + 72) & 4) != 0 && *(_QWORD *)(v2 + 48) )
    {
      v29 = 3;
      v30[0] = 0;
      sub_86C7A0(v2, (__int64)v30, &v29);
      if ( v30[0] )
        sub_685910(v30[0], (FILE *)v30);
      v2 = *(_QWORD *)(a1 + 40);
    }
    v10 = qword_4F5FD68;
    if ( qword_4F5FD68 != v2 )
    {
      do
      {
        v11 = *(_BYTE *)(v10 + 32);
        if ( (unsigned __int8)(v11 - 3) <= 1u )
          break;
        if ( v11 )
        {
          v12 = *(_QWORD **)(v10 + 8);
          if ( v11 != 1 || (*(_BYTE *)(v10 + 56) & 1) != 0 && *(_QWORD *)(*(_QWORD *)(v10 + 16) + 64LL) )
          {
            v10 = *(_QWORD *)(v10 + 8);
          }
          else
          {
            if ( v12 )
            {
              *v12 = *(_QWORD *)v10;
              v13 = *(_QWORD *)(v10 + 8);
            }
            else
            {
              qword_4F5FD70 = *(_QWORD *)v10;
              v13 = 0;
            }
            if ( *(_QWORD *)v10 )
              *(_QWORD *)(*(_QWORD *)v10 + 8LL) = v13;
            else
              qword_4F5FD68 = v13;
            *(_QWORD *)v10 = qword_4F5FD60;
            qword_4F5FD60 = v10;
            v10 = (__int64)v12;
          }
        }
        else
        {
          if ( (*(_BYTE *)(v10 + 72) & 1) != 0 || *(_QWORD *)(v10 + 48) )
            break;
          v10 = *(_QWORD *)(v10 + 8);
        }
      }
      while ( *(_QWORD *)(a1 + 40) != v10 );
      v2 = qword_4F5FD68;
    }
LABEL_8:
    *(_QWORD *)v2 = a1;
    *(_QWORD *)(a1 + 8) = qword_4F5FD68;
  }
  else
  {
    qword_4F5FD70 = a1;
  }
  result = &dword_4D047EC;
  qword_4F5FD68 = a1;
  if ( dword_4D047EC )
  {
    if ( *(_BYTE *)(a1 + 32) == 1 )
    {
      v20 = *(_QWORD *)(a1 + 40);
      if ( v20 )
      {
        if ( *(_BYTE *)(v20 + 40) == 22 )
        {
          v21 = *(_QWORD *)(a1 + 8);
          if ( v21 )
          {
LABEL_70:
            if ( *(_BYTE *)(v21 + 32) == 1 )
            {
              result = *(unsigned int **)(v21 + 40);
              if ( result )
              {
                if ( *((_BYTE *)result + 40) == 21 )
                {
                  v22 = *((_QWORD *)result + 9);
                  for ( i = *(_QWORD *)(*(_QWORD *)(v20 + 80) + 120LL); ; i = sub_8D4050(i) )
                  {
                    result = (unsigned int *)sub_8D3410(i);
                    if ( !(_DWORD)result )
                      break;
                    result = (unsigned int *)i;
                    if ( *(_BYTE *)(i + 140) == 12 )
                    {
                      if ( *(_QWORD *)(i + 8) )
                        return result;
                      do
                        result = (unsigned int *)*((_QWORD *)result + 20);
                      while ( *((_BYTE *)result + 140) == 12 );
                    }
                    v24 = *(_QWORD *)(v22 + 8);
                    if ( (unsigned int *)v24 != result )
                    {
                      if ( !v24 )
                        continue;
                      if ( !dword_4F07588 )
                        continue;
                      v25 = *(_QWORD *)(v24 + 32);
                      if ( *((_QWORD *)result + 4) != v25 || !v25 )
                        continue;
                    }
                    v28 = *(_QWORD **)(v21 + 8);
                    if ( v28 )
                    {
                      *v28 = *(_QWORD *)v21;
                      v28 = *(_QWORD **)(v21 + 8);
                    }
                    else
                    {
                      qword_4F5FD70 = *(_QWORD *)v21;
                    }
                    if ( *(_QWORD *)v21 )
                      *(_QWORD *)(*(_QWORD *)v21 + 8LL) = v28;
                    else
                      qword_4F5FD68 = (__int64)v28;
                    result = (unsigned int *)qword_4F5FD60;
                    *(_QWORD *)v21 = qword_4F5FD60;
                    qword_4F5FD60 = v21;
                    v21 = *(_QWORD *)(a1 + 8);
                    if ( v21 )
                      goto LABEL_70;
                    return result;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
