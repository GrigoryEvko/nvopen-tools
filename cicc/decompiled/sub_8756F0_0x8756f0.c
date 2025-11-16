// Function: sub_8756F0
// Address: 0x8756f0
//
__int64 __fastcall sub_8756F0(__int16 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v5; // rdx
  char v8; // cl
  char v9; // si
  int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // r15
  __m128i *v13; // rdi
  __int64 result; // rax
  __int64 v15; // rbx
  _BYTE *v16; // rbx
  char v17; // si
  _BYTE *v18; // r12
  __int64 v19; // rax
  _BOOL4 v20; // [rsp+4h] [rbp-4Ch]
  unsigned __int8 v22; // [rsp+1Bh] [rbp-35h] BYREF
  int v23[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = a1 & 0x200;
  if ( (a1 & 2) == 0 )
  {
    if ( (a1 & 0x800) != 0 )
    {
      v12 = 0;
      v10 = 0;
      v20 = v5 == 0;
      goto LABEL_15;
    }
    goto LABEL_26;
  }
  v8 = *(_BYTE *)(a2 + 81);
  v9 = v8 & 2;
  if ( (a1 & 0x800) == 0 && *(_BYTE *)(a2 + 80) == 9 && (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 176LL) & 1) != 0 )
  {
    if ( !v9 )
    {
      v20 = 0;
      v12 = 0;
      v10 = 1;
      *(_BYTE *)(a2 + 81) = v8 | 2;
      goto LABEL_15;
    }
    goto LABEL_26;
  }
  if ( v9 )
  {
    if ( dword_4F077C4 == 2 || *(_BYTE *)(a2 + 80) != 7 || (a1 & 0x200) != 0 )
    {
LABEL_26:
      v20 = 0;
      v12 = 0;
      v10 = 0;
      goto LABEL_15;
    }
    v20 = 1;
  }
  else
  {
    *(_BYTE *)(a2 + 81) |= 2u;
    v20 = v5 == 0;
  }
  v10 = 1;
  *(_QWORD *)(a2 + 48) = *a3;
  v11 = sub_87D520(a2);
  v12 = v11;
  if ( !v11 )
    goto LABEL_15;
  v13 = *(__m128i **)(v11 + 72);
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 80) - 19) <= 3u )
  {
    if ( v13 )
      goto LABEL_15;
    goto LABEL_8;
  }
  *(_QWORD *)(v11 + 64) = *a3;
  if ( !v13 )
  {
LABEL_8:
    v10 = 1;
    *(_QWORD *)(v11 + 72) = sub_7274B0(*(_BYTE *)(v11 - 8) & 1);
    goto LABEL_15;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 80) - 19) > 3u )
    sub_727480(v13);
LABEL_15:
  if ( !*(_DWORD *)(a2 + 44) )
    *(_DWORD *)(a2 + 44) = ++dword_4F066AC;
  if ( qword_4D04900 )
    sub_8754F0(a1, (unsigned __int8 *)a2, (__int64)a3);
  result = *(unsigned __int8 *)(a2 + 80);
  if ( (_BYTE)result == 12 )
  {
    if ( v10 )
      return result;
LABEL_28:
    if ( a1 < 0 )
      return result;
LABEL_29:
    result = dword_4F04C3C;
LABEL_30:
    if ( (_DWORD)result )
      return result;
    goto LABEL_31;
  }
  if ( (_BYTE)result == 18 )
    return result;
  if ( !v10 )
    goto LABEL_28;
  result = (unsigned int)(result - 4);
  v15 = (unsigned __int16)a1 & 0x8000;
  if ( (unsigned __int8)result > 1u )
  {
    if ( v15 )
      return result;
    result = dword_4F04C3C;
    if ( v12 )
    {
      if ( dword_4F04C3C )
        return result;
      *(_QWORD *)(v12 + 96) = 0;
      goto LABEL_31;
    }
    goto LABEL_30;
  }
  if ( !v12 )
  {
    if ( v15 )
      return result;
    goto LABEL_29;
  }
  result = *(_QWORD *)(v12 + 96);
  if ( v15 )
    return result;
  if ( !dword_4F04C3C )
  {
    *(_QWORD *)(v12 + 96) = 0;
    if ( !result )
      *(_BYTE *)(*(_QWORD *)(a2 + 96) + 180LL) |= 0x80u;
LABEL_31:
    result = sub_87D1A0(a2, &v22);
    v16 = (_BYTE *)result;
    if ( result )
    {
      if ( *(_DWORD *)a3 )
      {
        result = v22;
        if ( v22 != 11 || (v16[193] & 0x10) == 0 )
        {
          if ( v20 )
          {
            v17 = v22;
            v18 = v16;
          }
          else
          {
            if ( dword_4F07270[0] != dword_4F073B8[0] && (*(v16 - 8) & 1) != 0 )
            {
              sub_7296C0(v23);
              v18 = sub_727110();
              sub_729730(v23[0]);
            }
            else
            {
              v18 = sub_727110();
            }
            v19 = *a3;
            *((_QWORD *)v18 + 3) = v16;
            v17 = 53;
            *(_QWORD *)v18 = v19;
            LOBYTE(v19) = v22;
            v22 = 53;
            v18[16] = v19;
            if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 1 )
              v18[57] |= 8u;
          }
          return sub_8699D0((__int64)v18, v17, a4);
        }
      }
    }
    return result;
  }
  if ( !result )
  {
    result = *(_QWORD *)(a2 + 96);
    *(_BYTE *)(result + 180) |= 0x80u;
  }
  return result;
}
