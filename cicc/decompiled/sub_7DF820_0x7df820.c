// Function: sub_7DF820
// Address: 0x7df820
//
__int64 __fastcall sub_7DF820(__int64 a1, int *a2, unsigned int a3)
{
  __int64 result; // rax
  int v5; // edx
  char v6; // r14
  __int64 *v7; // r15
  __int64 *v8; // r14
  __int64 v9; // r14
  int v10; // [rsp+14h] [rbp-3Ch] BYREF
  int v11; // [rsp+18h] [rbp-38h] BYREF
  _DWORD v12[13]; // [rsp+1Ch] [rbp-34h] BYREF

  result = *(unsigned __int8 *)(a1 + 24);
  v10 = 0;
  if ( (_BYTE)result == 1 )
  {
    v6 = *(_BYTE *)(a1 + 56);
    result = sub_730FB0(v6);
    if ( (_DWORD)result )
    {
      if ( !a3 )
      {
        result = (__int64)sub_72BA30(5u);
        *(_QWORD *)a1 = result;
      }
      v5 = 1;
    }
    else
    {
      result = *(_QWORD *)(a1 + 72);
      v7 = *(__int64 **)(result + 16);
      if ( v6 != 91 )
      {
        if ( (unsigned __int8)(v6 - 103) <= 1u )
        {
          v9 = v7[2];
          sub_7DF820(*(_QWORD *)(result + 16), &v11, 1);
          result = sub_7DF820(v9, v12, 1);
          v5 = v11;
          if ( !v11 )
            goto LABEL_3;
          v5 = v12[0];
          if ( !v12[0] )
            goto LABEL_3;
          v10 = 1;
          v5 = a3;
          if ( a3 )
            goto LABEL_3;
          sub_7DF820(v7, &v11, 0);
          sub_7DF820(v9, v12, 0);
          result = *v7;
          *(_QWORD *)a1 = *v7;
        }
        v5 = v10;
        goto LABEL_3;
      }
      result = sub_7DF820(*(_QWORD *)(result + 16), &v10, a3);
      v5 = v10;
      if ( (a3 & 1) == 0 && v10 )
      {
        result = *v7;
        *(_QWORD *)a1 = *v7;
      }
    }
  }
  else
  {
    v5 = 0;
    if ( (_BYTE)result == 10 )
    {
      v8 = *(__int64 **)(a1 + 56);
      result = sub_7DF820(v8, &v10, a3);
      v5 = v10;
      if ( v10 )
      {
        if ( (a3 & 1) == 0 )
        {
          result = *v8;
          *(_QWORD *)a1 = *v8;
        }
      }
    }
  }
LABEL_3:
  *a2 = v5;
  return result;
}
