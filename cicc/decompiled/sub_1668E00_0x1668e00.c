// Function: sub_1668E00
// Address: 0x1668e00
//
void __fastcall sub_1668E00(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  _BYTE *v10; // rax
  const char *v11; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+10h] [rbp-30h]
  char v13; // [rsp+11h] [rbp-2Fh]

  sub_1668250((__int64 *)a1, a2 & 0xFFFFFFFFFFFFFFFBLL);
  v4 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(*(_QWORD *)(a2 - 24)) + 16) - 34;
  if ( (unsigned int)v4 <= 0x36 && (v5 = 0x40018000000001LL, _bittest64(&v5, v4)) )
  {
    sub_1665790((_BYTE *)a1, a2);
  }
  else
  {
    v6 = *(_QWORD *)a1;
    v13 = 1;
    v11 = "The unwind destination does not have an exception handling instruction!";
    v12 = 3;
    if ( v6 )
    {
      sub_16E2CE0(&v11, v6);
      v7 = *(_BYTE **)(v6 + 24);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
      {
        sub_16E7DE0(v6, 10);
      }
      else
      {
        *(_QWORD *)(v6 + 24) = v7 + 1;
        *v7 = 10;
      }
      v8 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 72) = 1;
      if ( v8 )
      {
        if ( *(_BYTE *)(a2 + 16) <= 0x17u )
        {
          sub_1553920((__int64 *)a2, v8, 1, a1 + 16);
          v9 = *(_QWORD *)a1;
          v10 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v10 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            goto LABEL_9;
        }
        else
        {
          sub_155BD40(a2, v8, a1 + 16, 0);
          v9 = *(_QWORD *)a1;
          v10 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v10 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
LABEL_9:
            *(_QWORD *)(v9 + 24) = v10 + 1;
            *v10 = 10;
            return;
          }
        }
        sub_16E7DE0(v9, 10);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 72) = 1;
    }
  }
}
