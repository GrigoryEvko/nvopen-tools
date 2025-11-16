// Function: sub_2E7AA20
// Address: 0x2e7aa20
//
__int64 __fastcall sub_2E7AA20(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // r8
  _QWORD *v4; // rdi
  _QWORD *v5; // rdx
  __int64 result; // rax
  _BYTE *v7; // rax
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF

  v3 = (_BYTE *)a1[75];
  v8 = a2;
  v4 = (_QWORD *)a1[74];
  if ( (unsigned int)((v3 - (_BYTE *)v4) >> 3) )
  {
    v5 = v4;
    LODWORD(result) = 0;
    while ( 1 )
    {
      result = (unsigned int)(result + 1);
      if ( *v5 == v8 )
        break;
      ++v5;
      if ( (unsigned int)((v3 - (_BYTE *)v4) >> 3) == (_DWORD)result )
        goto LABEL_6;
    }
  }
  else
  {
LABEL_6:
    if ( v3 == (_BYTE *)a1[76] )
    {
      sub_2E7A890((__int64)(a1 + 74), v3, &v8);
      v7 = (_BYTE *)a1[75];
      v4 = (_QWORD *)a1[74];
    }
    else
    {
      if ( v3 )
      {
        *(_QWORD *)v3 = v8;
        v3 = (_BYTE *)a1[75];
        v4 = (_QWORD *)a1[74];
      }
      v7 = v3 + 8;
      a1[75] = v3 + 8;
    }
    return (v7 - (_BYTE *)v4) >> 3;
  }
  return result;
}
