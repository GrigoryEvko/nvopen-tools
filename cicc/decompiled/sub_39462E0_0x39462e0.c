// Function: sub_39462E0
// Address: 0x39462e0
//
__int64 __fastcall sub_39462E0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  _BYTE *v5; // rdx
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  char v11; // si
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  char v18; // cl
  unsigned __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = a2;
  v4 = *(_BYTE **)(a2 + 16);
  v5 = *(_BYTE **)(a2 + 24);
  if ( (*a1 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
  {
    if ( (*(_BYTE *)a1 & 2) != 0 )
    {
      if ( v4 == v5 )
      {
        v3 = sub_16E7EE0(a2, "<", 1u);
      }
      else
      {
        *v5 = 60;
        ++*(_QWORD *)(a2 + 24);
      }
      v7 = sub_16E7AB0(v3, (unsigned __int16)(*a1 >> 2));
      v8 = *(_QWORD *)(v7 + 24);
      v9 = v7;
      if ( (unsigned __int64)(*(_QWORD *)(v7 + 16) - v8) <= 2 )
      {
        v9 = sub_16E7EE0(v7, " x ", 3u);
      }
      else
      {
        *(_BYTE *)(v8 + 2) = 32;
        *(_WORD *)v8 = 30752;
        *(_QWORD *)(v7 + 24) += 3LL;
      }
      v10 = *a1;
      v11 = *(_BYTE *)a1 & 2;
      v12 = *a1 >> 2;
      if ( (*(_BYTE *)a1 & 1) != 0 )
      {
        v13 = v10 >> 18;
        if ( v11 )
        {
          v12 = (unsigned __int16)v13;
          LOWORD(v13) = v10 >> 34;
        }
        else
        {
          v12 = (unsigned __int16)v12;
        }
        v14 = (unsigned __int16)v13;
        v15 = 1;
        v16 = v12 | (v14 << 16);
      }
      else
      {
        v16 = v10 >> 18;
        if ( !v11 )
          LODWORD(v16) = *a1 >> 2;
        v15 = 0;
        v16 = (unsigned int)v16;
      }
      v23[0] = v15 | (4 * v16);
      sub_39462E0(v23, v9, v15, v12);
      result = *(_QWORD *)(v9 + 24);
      if ( *(_QWORD *)(v9 + 16) == result )
      {
        return sub_16E7EE0(v9, ">", 1u);
      }
      else
      {
        *(_BYTE *)result = 62;
        ++*(_QWORD *)(v9 + 24);
      }
    }
    else
    {
      if ( (*(_BYTE *)a1 & 3) == 1 )
      {
        if ( v4 == v5 )
        {
          v3 = sub_16E7EE0(a2, "p", 1u);
        }
        else
        {
          *v5 = 112;
          ++*(_QWORD *)(a2 + 24);
        }
        v22 = *a1;
        if ( (*(_BYTE *)a1 & 2) != 0 )
          v20 = (v22 >> 34) & 0x7FFFFF;
        else
          v20 = (v22 >> 18) & 0x7FFFFF;
      }
      else
      {
        if ( v4 == v5 )
        {
          v3 = sub_16E7EE0(a2, "s", 1u);
        }
        else
        {
          *v5 = 115;
          ++*(_QWORD *)(a2 + 24);
        }
        v17 = *a1;
        v18 = *(_BYTE *)a1 & 1;
        v19 = *a1 >> 2;
        if ( (*(_BYTE *)a1 & 2) != 0 )
        {
          v21 = (unsigned int)(v17 >> 18);
          v20 = (unsigned __int16)(v17 >> 18);
          if ( !v18 )
            v20 = v21;
        }
        else
        {
          v20 = (unsigned int)v19;
          if ( v18 )
            v20 = (unsigned __int16)v19;
        }
      }
      return sub_16E7A90(v3, v20);
    }
  }
  else if ( (unsigned __int64)(v4 - v5) <= 0xA )
  {
    return sub_16E7EE0(a2, "LLT_invalid", 0xBu);
  }
  else
  {
    qmemcpy(v5, "LLT_invalid", 11);
    *(_QWORD *)(a2 + 24) += 11LL;
    return 26988;
  }
  return result;
}
