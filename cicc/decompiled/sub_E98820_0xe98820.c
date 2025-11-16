// Function: sub_E98820
// Address: 0xe98820
//
void (*__fastcall sub_E98820(_QWORD *a1, __int64 a2, _QWORD *a3))()
{
  char v4; // al
  void (*result)(); // rax
  __int64 v6; // rdi
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdi
  void *v13; // rax
  _QWORD v14[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v15; // [rsp+20h] [rbp-60h]
  _QWORD v16[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v17; // [rsp+50h] [rbp-30h]

  v4 = *(_BYTE *)(a2 + 8);
  if ( (v4 & 4) != 0 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 0x70) == 0x20 )
    {
      *(_WORD *)(a2 + 8) &= 0x8FFBu;
      *(_QWORD *)(a2 + 24) = 0;
      *(_QWORD *)a2 = 0;
    }
    else
    {
      *(_QWORD *)a2 = 0;
      *(_BYTE *)(a2 + 8) = v4 & 0xFB;
    }
    goto LABEL_4;
  }
  if ( !*(_QWORD *)a2 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 )
      goto LABEL_4;
    if ( v4 >= 0 )
    {
      v12 = *(_QWORD *)(a2 + 24);
      *(_BYTE *)(a2 + 8) = v4 | 8;
      v13 = sub_E807D0(v12);
      *(_QWORD *)a2 = v13;
      if ( !v13 && (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 )
      {
LABEL_4:
        result = (void (*)())(*(_QWORD *)(a1[36] + 8LL) + 56LL);
        *(_QWORD *)a2 = result;
        v6 = a1[2];
        if ( v6 )
        {
          result = *(void (**)())(*(_QWORD *)v6 + 16LL);
          if ( result != nullsub_340 )
            return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(v6, a2);
        }
        return result;
      }
      v4 = *(_BYTE *)(a2 + 8);
    }
  }
  v8 = a1[1];
  if ( (v4 & 1) != 0 )
  {
    v9 = *(__int64 **)(a2 - 8);
    v10 = *v9;
    v11 = v9 + 3;
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  v14[2] = v11;
  v14[3] = v10;
  v15 = 1283;
  v16[0] = v14;
  v17 = 770;
  v14[0] = "symbol '";
  v16[2] = "' is already defined";
  return (void (*)())sub_E66880(v8, a3, (__int64)v16);
}
