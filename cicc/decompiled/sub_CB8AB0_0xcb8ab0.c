// Function: sub_CB8AB0
// Address: 0xcb8ab0
//
signed __int64 __fastcall sub_CB8AB0(__int64 a1, __int64 a2)
{
  int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // r13
  signed __int64 result; // rax
  signed __int64 v6; // rdx
  __int64 v7; // rcx
  _BYTE *v8; // r12
  __int64 v9; // rdx
  signed __int64 v10; // rsi
  int v11; // r8d
  char v12; // al
  __int64 v13; // r14
  __int64 v14; // r13
  char v15; // [rsp+Dh] [rbp-23h] BYREF
  __int16 v16; // [rsp+Eh] [rbp-22h] BYREF

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 56);
  v4 = *(_QWORD *)(v3 + 88);
  if ( (*(_BYTE *)(v3 + 40) & 2) == 0 || !isalpha((unsigned __int8)a2) )
    goto LABEL_3;
  if ( isupper((unsigned __int8)a2) )
  {
    v12 = tolower((unsigned __int8)a2);
  }
  else
  {
    v11 = islower((unsigned __int8)a2);
    v12 = a2;
    if ( v11 )
      v12 = toupper((unsigned __int8)a2);
  }
  if ( v12 != (_DWORD)a2 )
  {
    v13 = *(_QWORD *)a1;
    v14 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)a1 = &v15;
    *(_QWORD *)(a1 + 8) = (char *)&v16 + 1;
    v15 = a2;
    v16 = 93;
    result = sub_CB7E40(a1, a2);
    *(_QWORD *)a1 = v13;
    *(_QWORD *)(a1 + 8) = v14;
  }
  else
  {
LABEL_3:
    result = *(unsigned int *)(a1 + 16);
    if ( !(_DWORD)result )
    {
      result = *(_QWORD *)(a1 + 40);
      v6 = *(_QWORD *)(a1 + 32);
      if ( result >= v6 )
      {
        v10 = (v6 + 1) / 2 + ((v6 + 1 + ((unsigned __int64)(v6 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
        if ( v6 < v10 )
        {
          sub_CB7740(a1, v10);
          result = *(_QWORD *)(a1 + 40);
        }
      }
      v7 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = result + 1;
      *(_QWORD *)(v7 + 8 * result) = (unsigned __int8)v2 | 0x10000000LL;
    }
    v8 = (_BYTE *)(v4 + v2);
    if ( !*v8 )
    {
      v9 = *(_QWORD *)(a1 + 56);
      result = *(unsigned int *)(v9 + 84);
      *(_DWORD *)(v9 + 84) = result + 1;
      *v8 = result;
    }
  }
  return result;
}
