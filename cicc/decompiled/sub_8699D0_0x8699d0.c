// Function: sub_8699D0
// Address: 0x8699d0
//
__int64 __fastcall sub_8699D0(__int64 a1, char a2, _QWORD *a3)
{
  char v3; // r15
  _QWORD *v5; // rbx
  __int64 result; // rax
  _BYTE *v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // rax
  _BOOL4 v10; // ecx
  __int64 v11; // rdx
  __int64 v12; // rdx
  int v13[13]; // [rsp+Ch] [rbp-34h] BYREF

  v3 = a2;
  v5 = (_QWORD *)a1;
  result = dword_4F073B8[0];
  if ( dword_4F07270[0] == dword_4F073B8[0] )
  {
    if ( a3 )
    {
      v7 = a3;
    }
    else
    {
      result = (__int64)sub_727090();
      v7 = (_BYTE *)result;
    }
    v7[16] = a2;
    *((_QWORD *)v7 + 3) = a1;
    if ( a2 != 53 )
    {
      if ( a2 != 21 )
        goto LABEL_7;
      goto LABEL_14;
    }
LABEL_21:
    v3 = *(_BYTE *)(a1 + 16);
    v5 = *(_QWORD **)(a1 + 24);
    if ( v3 != 21 )
      goto LABEL_7;
LABEL_14:
    v5[7] = v7;
    if ( a3 )
      return result;
    return sub_869970((__int64)v7);
  }
  if ( a2 == 21 )
  {
    v7 = a3;
    if ( !a3 )
    {
      result = (__int64)sub_727090();
      v7 = (_BYTE *)result;
    }
    v7[16] = 21;
    *((_QWORD *)v7 + 3) = a1;
    goto LABEL_14;
  }
  if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
  {
    sub_7296C0(v13);
    if ( a3 )
    {
      v7 = a3;
      if ( (*(_BYTE *)(a3 - 1) & 1) == 0 )
      {
        v7 = sub_727090();
        v8 = (_QWORD *)a3[1];
        if ( v8 )
        {
          *((_QWORD *)v7 + 1) = v8;
          *v8 = v7;
        }
        else
        {
          *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328) = v7;
        }
        v9 = *a3;
        if ( *a3 )
        {
          *(_QWORD *)v7 = v9;
          *(_QWORD *)(v9 + 8) = v7;
        }
        else
        {
          *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336) = v7;
        }
      }
    }
    else
    {
      v7 = sub_727090();
    }
    result = (__int64)sub_729730(v13[0]);
  }
  else if ( a3 )
  {
    v7 = a3;
  }
  else
  {
    result = (__int64)sub_727090();
    v7 = (_BYTE *)result;
  }
  v7[16] = a2;
  *((_QWORD *)v7 + 3) = a1;
  if ( a2 == 53 )
    goto LABEL_21;
LABEL_7:
  result = sub_72A270((__int64)v5, v3);
  if ( result )
  {
    if ( !*(_QWORD *)(result + 96)
      && (dword_4F04C58 == -1
       || (*(v7 - 8) & 1) == 0
       || (*(_BYTE *)(result + 89) & 4) != 0
       || (*(_BYTE *)(result + 88) & 0x70) == 0
       || ((v3 - 7) & 0xFB) != 0) )
    {
      if ( v3 != 6 )
        goto LABEL_39;
      if ( dword_4F04C44 == -1 )
      {
        v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v12 + 6) & 6) == 0 && *(_BYTE *)(v12 + 4) != 12 )
          goto LABEL_39;
      }
      v10 = 1;
      if ( (unsigned __int8)(*(_BYTE *)(result + 140) - 9) <= 2u )
        v10 = (*(_BYTE *)(result + 177) & 0xA0) != 32;
      if ( dword_4D047B0 && !dword_4D047AC
        || (v11 = 776LL * dword_4F04C64, *(_BYTE *)(qword_4F04C68[0] + v11 + 4) != 1)
        || *(_BYTE *)(qword_4F04C68[0] + v11 - 772) != 8 )
      {
        if ( v10 )
LABEL_39:
          *(_QWORD *)(result + 96) = v7;
      }
    }
  }
  else if ( v3 == 29 )
  {
    v5[8] = v7;
  }
  else if ( v3 == 58 )
  {
    v5[5] = v7;
  }
  if ( !a3 )
    return sub_869970((__int64)v7);
  return result;
}
