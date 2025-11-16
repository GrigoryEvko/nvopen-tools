// Function: sub_BF9240
// Address: 0xbf9240
//
void __fastcall sub_BF9240(_BYTE *a1, __int64 a2)
{
  int v2; // eax
  __int64 *v3; // rdx
  __int64 v4; // r14
  const char *v5; // rax
  __int64 v6; // r15
  _BYTE *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8
  _BYTE *v10; // rax
  const char *v11; // [rsp+0h] [rbp-50h] BYREF
  char v12; // [rsp+20h] [rbp-30h]
  char v13; // [rsp+21h] [rbp-2Fh]

  v2 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v3 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 24LL) + 16LL);
  v4 = *v3;
  if ( *(_BYTE *)(*v3 + 8) != 7 )
  {
    if ( v2 != 1 || v4 != *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL) )
    {
      v13 = 1;
      v5 = "Function return type does not match operand type of return inst!";
      goto LABEL_5;
    }
LABEL_13:
    sub_BF90E0(a1, a2);
    return;
  }
  if ( !v2 )
    goto LABEL_13;
  v13 = 1;
  v5 = "Found return instr that returns non-void in Function of void return type!";
LABEL_5:
  v6 = *(_QWORD *)a1;
  v11 = v5;
  v12 = 3;
  if ( v6 )
  {
    sub_CA0E80(&v11, v6);
    v7 = *(_BYTE **)(v6 + 32);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 24) )
    {
      sub_CB5D20(v6, 10);
    }
    else
    {
      *(_QWORD *)(v6 + 32) = v7 + 1;
      *v7 = 10;
    }
    v8 = *(_QWORD *)a1;
    a1[152] = 1;
    if ( v8 )
    {
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
      v9 = *(_QWORD *)a1;
      v10 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        v9 = sub_CB5D20(*(_QWORD *)a1, 32);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = v10 + 1;
        *v10 = 32;
      }
      sub_A587F0(v4, v9, 0, 0);
    }
  }
  else
  {
    a1[152] = 1;
  }
}
