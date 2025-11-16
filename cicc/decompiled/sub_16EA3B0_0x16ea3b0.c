// Function: sub_16EA3B0
// Address: 0x16ea3b0
//
signed __int64 __fastcall sub_16EA3B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // r13
  signed __int64 result; // rax
  signed __int64 v10; // rdx
  __int64 v11; // rcx
  _BYTE *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  signed __int64 v15; // rsi
  __int64 v16; // rcx
  char v17; // al
  __int64 v18; // r14
  __int64 v19; // r13
  char v20; // [rsp+Dh] [rbp-23h] BYREF
  __int16 v21; // [rsp+Eh] [rbp-22h] BYREF

  v6 = a2;
  v7 = *(_QWORD *)(a1 + 56);
  v8 = *(_QWORD *)(v7 + 88);
  if ( (*(_BYTE *)(v7 + 40) & 2) == 0 || !isalpha((unsigned __int8)a2) )
    goto LABEL_3;
  if ( isupper((unsigned __int8)a2) )
  {
    v17 = tolower((unsigned __int8)a2);
  }
  else
  {
    a5 = islower((unsigned __int8)a2);
    v17 = a2;
    if ( a5 )
      v17 = toupper((unsigned __int8)a2);
  }
  if ( v17 != (_DWORD)a2 )
  {
    v18 = *(_QWORD *)a1;
    v19 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)a1 = &v20;
    *(_QWORD *)(a1 + 8) = (char *)&v21 + 1;
    v20 = a2;
    v21 = 93;
    result = sub_16E97A0(a1, a2, 93, v16, a5, a6);
    *(_QWORD *)a1 = v18;
    *(_QWORD *)(a1 + 8) = v19;
  }
  else
  {
LABEL_3:
    result = *(unsigned int *)(a1 + 16);
    if ( !(_DWORD)result )
    {
      result = *(_QWORD *)(a1 + 40);
      v10 = *(_QWORD *)(a1 + 32);
      if ( result >= v10 )
      {
        v14 = (v10 + 1) / 2;
        v15 = v14 + ((v10 + 1 + ((unsigned __int64)(v10 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
        if ( v10 < v15 )
        {
          sub_16E90A0(a1, v15, v10, v14, a5, a6);
          result = *(_QWORD *)(a1 + 40);
        }
      }
      v11 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = result + 1;
      *(_QWORD *)(v11 + 8 * result) = (unsigned __int8)v6 | 0x10000000LL;
    }
    v12 = (_BYTE *)(v8 + v6);
    if ( !*v12 )
    {
      v13 = *(_QWORD *)(a1 + 56);
      result = *(unsigned int *)(v13 + 84);
      *(_DWORD *)(v13 + 84) = result + 1;
      *v12 = result;
    }
  }
  return result;
}
