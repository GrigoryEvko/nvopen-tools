// Function: sub_AAD9D0
// Address: 0xaad9d0
//
__int64 __fastcall sub_AAD9D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // ebx
  __int64 result; // rax
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-18h]

  v2 = *(unsigned int *)(a1 + 8);
  v6 = v2;
  v3 = a2 - v2;
  if ( (unsigned int)v2 > 0x40 )
  {
    sub_C43690(&v5, 0, 0);
    v2 = v6;
    a2 = v6 + v3;
    if ( v6 == (_DWORD)a2 )
      goto LABEL_6;
    if ( (unsigned int)a2 <= 0x3F )
      goto LABEL_4;
  }
  else
  {
    v5 = 0;
    if ( (_DWORD)v2 == (_DWORD)a2 )
    {
      result = 0;
      goto LABEL_8;
    }
    if ( (unsigned int)a2 <= 0x3F )
    {
LABEL_4:
      if ( (unsigned int)v2 <= 0x40 )
      {
        v5 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 + 64) << a2;
        goto LABEL_6;
      }
    }
  }
  sub_C43C90(&v5, a2, v2);
LABEL_6:
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
    result = v5;
    LODWORD(a2) = v6;
LABEL_8:
    *(_QWORD *)a1 &= result;
    goto LABEL_9;
  }
  result = sub_C43B90(a1, &v5);
  LODWORD(a2) = v6;
LABEL_9:
  if ( (unsigned int)a2 > 0x40 )
  {
    if ( v5 )
      return j_j___libc_free_0_0(v5);
  }
  return result;
}
