// Function: sub_87D9B0
// Address: 0x87d9b0
//
__int64 __fastcall sub_87D9B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        FILE *a4,
        __int64 a5,
        char a6,
        unsigned int a7,
        _DWORD *a8)
{
  _DWORD *v13; // rdi
  __int64 v14; // rax
  char v15; // r9
  char v16; // r8
  char v17; // r8
  int v18; // r9d
  char v19; // r15
  __int64 result; // rax
  __int64 v21; // r14
  __int64 v22; // r9
  __int64 flags; // r10
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 *v26; // rdx
  __int64 v27; // [rsp-10h] [rbp-70h]
  FILE *v28; // [rsp+8h] [rbp-58h]
  char v29; // [rsp+1Eh] [rbp-42h]
  char v30; // [rsp+1Fh] [rbp-41h]
  _DWORD v31[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v13 = a8;
  v14 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v15 = *(_BYTE *)(v14 + 12);
  v16 = *(_BYTE *)(v14 + 13) & 0x40;
  if ( (v15 & 0x10) != 0 || qword_4D03C50 && *(char *)(qword_4D03C50 + 18LL) < 0 )
  {
    if ( v16 )
    {
      a6 = 5;
    }
    else if ( !dword_4F07730 )
    {
      a6 = 8;
    }
    v18 = v15 & 1;
    result = (__int64)v31;
    v31[0] = 0;
    if ( !a8 )
      v13 = v31;
  }
  else
  {
    if ( v16 )
      a6 = 5;
    v17 = v15 & 1;
    v18 = v15 & 1;
    v19 = *(_BYTE *)(v14 + 13) & 1;
    result = dword_4F04C40;
    if ( dword_4F04C40 != -1 )
    {
      v21 = qword_4F04C68[0] + 776LL * (int)dword_4F04C40;
      if ( (*(_BYTE *)(v21 + 7) & 8) != 0 )
      {
        result = *(_QWORD *)(v21 + 456);
        if ( result )
        {
          while ( 1 )
          {
            if ( *(_QWORD *)(result + 8) == a1
              && *(_QWORD *)(result + 16) == a2
              && *(_QWORD *)(result + 24) == a3
              && *(_DWORD *)(result + 40) == dword_4F06650[0]
              && *(_BYTE *)(result + 44) == a6
              && *(_DWORD *)(result + 48) == a7
              && v17 == *(_BYTE *)(result + 52)
              && v19 == *(_BYTE *)(result + 53) )
            {
              v22 = *(unsigned int *)(result + 32);
              flags = (unsigned int)a4->_flags;
              if ( (_DWORD)v22 == (_DWORD)flags )
              {
                v22 = *(unsigned __int16 *)(result + 36);
                flags = *((unsigned __int16 *)&a4->_flags + 2);
              }
              if ( v22 == flags )
                break;
            }
            result = *(_QWORD *)result;
            if ( !result )
              goto LABEL_36;
          }
        }
        else
        {
LABEL_36:
          result = qword_4F60008;
          if ( qword_4F60008 )
          {
            qword_4F60008 = *(_QWORD *)qword_4F60008;
          }
          else
          {
            v29 = a6;
            v28 = a4;
            v30 = v17;
            result = sub_823970(56);
            a6 = v29;
            a4 = v28;
            v17 = v30;
          }
          *(_QWORD *)result = 0;
          *(_DWORD *)(result + 40) = 0;
          v24 = *(_QWORD *)&dword_4F063F8;
          *(_BYTE *)(result + 44) = 3;
          *(_DWORD *)(result + 48) = 0;
          *(_QWORD *)(result + 32) = v24;
          *(_BYTE *)(result + 52) = 0;
          *(_QWORD *)(result + 8) = a1;
          *(_QWORD *)(result + 16) = a2;
          v25 = *(_QWORD *)&a4->_flags;
          *(_QWORD *)(result + 24) = a3;
          *(_QWORD *)(result + 32) = v25;
          LODWORD(v25) = dword_4F06650[0];
          *(_BYTE *)(result + 44) = a6;
          *(_DWORD *)(result + 40) = v25;
          *(_DWORD *)(result + 48) = a7;
          *(_BYTE *)(result + 52) = v17;
          *(_BYTE *)(result + 53) = v19;
          if ( !*(_QWORD *)(v21 + 456) )
            *(_QWORD *)(v21 + 456) = result;
          v26 = *(__int64 **)(v21 + 464);
          if ( v26 )
            *v26 = result;
          *(_QWORD *)(v21 + 464) = result;
        }
        return result;
      }
    }
    v31[0] = 0;
  }
  if ( a5 )
  {
    if ( (*(_BYTE *)(a5 + 17) & 1) != 0 )
      return result;
    result = sub_8774F0(a1, a3, a4, a6, a7, v18, v13);
    *(_BYTE *)(a5 + 17) |= 1u;
  }
  else
  {
    sub_8774F0(a1, a3, a4, a6, a7, v18, v13);
    result = v27;
  }
  if ( v31[0] )
  {
    result = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 19LL) |= 1u;
  }
  return result;
}
