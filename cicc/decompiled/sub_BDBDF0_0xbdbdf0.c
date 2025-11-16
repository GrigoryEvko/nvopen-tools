// Function: sub_BDBDF0
// Address: 0xbdbdf0
//
char *__fastcall sub_BDBDF0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rdx
  char *result; // rax
  __int64 v6; // rbx
  __int64 v7; // r8
  _BYTE *v8; // rax
  _QWORD v9[4]; // [rsp+0h] [rbp-60h] BYREF
  char v10; // [rsp+20h] [rbp-40h]
  char v11; // [rsp+21h] [rbp-3Fh]

  v9[0] = sub_9208B0(*(_QWORD *)(a1 + 136), a2);
  v9[1] = v4;
  result = (char *)sub_CA1930(v9);
  if ( (unsigned int)result <= 7 )
  {
    v11 = 1;
    result = "atomic memory access' size must be byte-sized";
  }
  else
  {
    if ( ((unsigned int)result & ((_DWORD)result - 1)) == 0 )
      return result;
    v11 = 1;
    result = "atomic memory access' operand must have a power-of-two size";
  }
  v6 = *(_QWORD *)a1;
  v9[0] = result;
  v10 = 3;
  if ( v6 )
  {
    sub_CA0E80(v9, v6);
    result = *(char **)(v6 + 32);
    if ( (unsigned __int64)result >= *(_QWORD *)(v6 + 24) )
    {
      result = (char *)sub_CB5D20(v6, 10);
    }
    else
    {
      *(_QWORD *)(v6 + 32) = result + 1;
      *result = 10;
    }
    v7 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( v7 )
    {
      if ( a2 )
      {
        v8 = *(_BYTE **)(v7 + 32);
        if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
        {
          v7 = sub_CB5D20(v7, 32);
        }
        else
        {
          *(_QWORD *)(v7 + 32) = v8 + 1;
          *v8 = 32;
        }
        result = (char *)sub_A587F0(a2, v7, 0, 0);
      }
      if ( a3 )
        return sub_BDBD80(a1, a3);
    }
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
  return result;
}
