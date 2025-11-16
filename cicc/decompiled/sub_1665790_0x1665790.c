// Function: sub_1665790
// Address: 0x1665790
//
void __fastcall sub_1665790(_BYTE *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r12
  _BYTE *v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdi
  _BYTE *v8; // rax
  const char *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 40);
  if ( a2 == sub_157EBA0(v3) )
  {
    sub_1663F80((__int64)a1, a2);
  }
  else
  {
    v4 = *(_QWORD *)a1;
    v11 = 1;
    v9 = "Terminator found in the middle of a basic block!";
    v10 = 3;
    if ( v4 )
    {
      sub_16E2CE0(&v9, v4);
      v5 = *(_BYTE **)(v4 + 24);
      if ( (unsigned __int64)v5 >= *(_QWORD *)(v4 + 16) )
      {
        sub_16E7DE0(v4, 10);
      }
      else
      {
        *(_QWORD *)(v4 + 24) = v5 + 1;
        *v5 = 10;
      }
      v6 = *(_QWORD *)a1;
      a1[72] = 1;
      if ( v3 && v6 )
      {
        if ( *(_BYTE *)(v3 + 16) <= 0x17u )
        {
          sub_1553920((__int64 *)v3, v6, 1, (__int64)(a1 + 16));
          v7 = *(_QWORD *)a1;
          v8 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v8 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            goto LABEL_9;
        }
        else
        {
          sub_155BD40(v3, v6, (__int64)(a1 + 16), 0);
          v7 = *(_QWORD *)a1;
          v8 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v8 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
LABEL_9:
            *(_QWORD *)(v7 + 24) = v8 + 1;
            *v8 = 10;
            return;
          }
        }
        sub_16E7DE0(v7, 10);
      }
    }
    else
    {
      a1[72] = 1;
    }
  }
}
