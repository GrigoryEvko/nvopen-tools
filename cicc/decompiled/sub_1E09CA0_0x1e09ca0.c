// Function: sub_1E09CA0
// Address: 0x1e09ca0
//
__int64 __fastcall sub_1E09CA0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  char *v4; // r14
  unsigned int v6; // ebx
  size_t v7; // rax
  __int64 v8; // rcx
  size_t v9; // rdx
  __int64 v10; // r8
  char *v11; // r14
  size_t v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  char *v17; // r14
  unsigned int v18; // ecx
  __int64 v19; // rsi
  __int64 v20; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v4 = (char *)byte_3F871B3;
    v6 = 0;
    do
    {
      result = *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v6 >> 6)) & (1LL << v6);
      if ( result )
      {
        v7 = strlen(v4);
        v8 = *(_QWORD *)(a2 + 24);
        v9 = v7;
        if ( v7 > *(_QWORD *)(a2 + 16) - v8 )
        {
          v10 = sub_16E7EE0(a2, v4, v7);
        }
        else
        {
          v10 = a2;
          if ( v7 )
          {
            if ( (_DWORD)v7 )
            {
              v13 = 0;
              do
              {
                v14 = v13++;
                *(_BYTE *)(v8 + v14) = v4[v14];
              }
              while ( v13 < (unsigned int)v9 );
            }
            *(_QWORD *)(a2 + 24) += v9;
            v10 = a2;
          }
        }
        switch ( v6 )
        {
          case 0u:
            v11 = "IsSSA";
            break;
          case 1u:
            v11 = "NoPHIs";
            break;
          case 2u:
            v11 = "TracksLiveness";
            break;
          case 3u:
            v11 = "NoVRegs";
            break;
          case 4u:
            v11 = "FailedISel";
            break;
          case 5u:
            v11 = "Legalized";
            break;
          case 6u:
            v11 = "RegBankSelected";
            break;
          case 7u:
            v11 = (char *)"Selected";
            break;
        }
        v20 = v10;
        v12 = strlen(v11);
        result = *(_QWORD *)(v20 + 24);
        if ( v12 <= *(_QWORD *)(v20 + 16) - result )
        {
          if ( (unsigned int)v12 < 8 )
          {
            if ( (v12 & 4) != 0 )
            {
              *(_DWORD *)result = *(_DWORD *)v11;
              *(_DWORD *)(result + (unsigned int)v12 - 4) = *(_DWORD *)&v11[(unsigned int)v12 - 4];
            }
            else if ( (_DWORD)v12 )
            {
              *(_BYTE *)result = *v11;
              if ( (v12 & 2) != 0 )
                *(_WORD *)(result + (unsigned int)v12 - 2) = *(_WORD *)&v11[(unsigned int)v12 - 2];
            }
          }
          else
          {
            v15 = (result + 8) & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)result = *(_QWORD *)v11;
            *(_QWORD *)(result + (unsigned int)v12 - 8) = *(_QWORD *)&v11[(unsigned int)v12 - 8];
            v16 = result - v15;
            v17 = &v11[-v16];
            result = ((_DWORD)v12 + (_DWORD)v16) & 0xFFFFFFF8;
            if ( (unsigned int)result >= 8 )
            {
              result = (unsigned int)result & 0xFFFFFFF8;
              v18 = 0;
              do
              {
                v19 = v18;
                v18 += 8;
                *(_QWORD *)(v15 + v19) = *(_QWORD *)&v17[v19];
              }
              while ( v18 < (unsigned int)result );
            }
          }
          *(_QWORD *)(v20 + 24) += v12;
        }
        else
        {
          result = sub_16E7EE0(v20, v11, v12);
        }
        v4 = ", ";
      }
      ++v6;
    }
    while ( *(_DWORD *)(a1 + 16) > v6 );
  }
  return result;
}
