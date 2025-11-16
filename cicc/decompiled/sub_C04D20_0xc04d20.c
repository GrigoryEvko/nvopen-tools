// Function: sub_C04D20
// Address: 0xc04d20
//
void __fastcall sub_C04D20(__int64 a1, _BYTE *a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdi
  _BYTE *v11; // rax
  const char *v12; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  sub_BFC6A0((__int64 *)a1, (__int64)a2);
  v4 = sub_AA4FF0(*((_QWORD *)a2 - 8));
  if ( !v4 )
    BUG();
  v5 = (unsigned int)*(unsigned __int8 *)(v4 - 24) - 39;
  if ( (unsigned int)v5 <= 0x38 && (v6 = 0x100060000000001LL, _bittest64(&v6, v5)) )
  {
    sub_BF90E0((_BYTE *)a1, (__int64)a2);
  }
  else
  {
    v7 = *(_QWORD *)a1;
    v14 = 1;
    v12 = "The unwind destination does not have an exception handling instruction!";
    v13 = 3;
    if ( v7 )
    {
      sub_CA0E80(&v12, v7);
      v8 = *(_BYTE **)(v7 + 32);
      if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
      {
        sub_CB5D20(v7, 10);
      }
      else
      {
        *(_QWORD *)(v7 + 32) = v8 + 1;
        *v8 = 10;
      }
      v9 = *(_BYTE **)a1;
      *(_BYTE *)(a1 + 152) = 1;
      if ( v9 )
      {
        if ( *a2 <= 0x1Cu )
        {
          sub_A5C020(a2, (__int64)v9, 1, a1 + 16);
          v10 = *(_QWORD *)a1;
          v11 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v11 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            goto LABEL_10;
        }
        else
        {
          sub_A693B0((__int64)a2, v9, a1 + 16, 0);
          v10 = *(_QWORD *)a1;
          v11 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v11 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
LABEL_10:
            *(_QWORD *)(v10 + 32) = v11 + 1;
            *v11 = 10;
            return;
          }
        }
        sub_CB5D20(v10, 10);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 152) = 1;
    }
  }
}
