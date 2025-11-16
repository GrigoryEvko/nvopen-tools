// Function: sub_9B6C80
// Address: 0x9b6c80
//
char __fastcall sub_9B6C80(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  _BYTE *v3; // rbx
  char *v4; // rcx
  char v5; // al
  unsigned __int64 v6; // rax
  int v7; // r13d
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  char result; // al
  int v12; // eax
  unsigned __int64 v13; // r13
  unsigned int v14; // r12d
  unsigned int v15; // r12d
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 *v18; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-28h]

  v2 = a2;
  v3 = (_BYTE *)*a2;
  if ( *(_BYTE *)*a2 != 17 )
    goto LABEL_4;
  v4 = (char *)a2[3];
  v5 = *v4;
  if ( (unsigned __int8)*v4 <= 0x1Cu )
  {
    if ( v5 == 5 && (unsigned __int16)(*((_WORD *)v4 + 1) - 25) <= 2u )
      goto LABEL_13;
LABEL_4:
    sub_D19B10(&v18, *a1, v2);
    if ( v19 <= 0x40 )
    {
      v10 = 1;
      if ( v18 )
      {
        _BitScanReverse64(&v17, (unsigned __int64)v18);
        v7 = 64 - (v17 ^ 0x3F);
        v8 = v7;
LABEL_7:
        if ( v7 == 1 )
        {
          v10 = 1;
        }
        else
        {
          _BitScanReverse64(&v9, v8 - 1);
          v10 = 1LL << (64 - ((unsigned __int8)v9 ^ 0x3Fu));
        }
      }
    }
    else
    {
      if ( *v18 )
      {
        _BitScanReverse64(&v6, *v18);
        v7 = 64 - (v6 ^ 0x3F);
        v8 = v7;
        j_j___libc_free_0_0(v18);
        goto LABEL_7;
      }
      j_j___libc_free_0_0(v18);
      v10 = 1;
    }
    return a1[1] < v10;
  }
  if ( (unsigned __int8)(v5 - 54) > 2u )
    goto LABEL_4;
LABEL_13:
  v12 = sub_BD2910(a2);
  v2 = a2;
  if ( v12 != 1 )
    goto LABEL_4;
  v13 = a1[1];
  v14 = *((_DWORD *)v3 + 8);
  if ( v14 <= 0x40 )
  {
    v16 = *((_QWORD *)v3 + 3);
    return v13 <= v16;
  }
  v15 = v14 - sub_C444A0(v3 + 24);
  result = 1;
  if ( v15 <= 0x40 )
  {
    v16 = **((_QWORD **)v3 + 3);
    return v13 <= v16;
  }
  return result;
}
