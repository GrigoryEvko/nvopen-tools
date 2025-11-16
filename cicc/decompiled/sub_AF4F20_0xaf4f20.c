// Function: sub_AF4F20
// Address: 0xaf4f20
//
__int64 __fastcall sub_AF4F20(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v5; // [rsp+0h] [rbp-8h]

  v1 = *(__int64 **)(a1 + 16);
  v2 = (__int64)(*(_QWORD *)(a1 + 24) - (_QWORD)v1) >> 3;
  if ( (unsigned int)(v2 - 2) > 1 && (_DWORD)v2 != 6 )
    goto LABEL_5;
  v3 = *v1;
  if ( *v1 == 17 )
  {
    if ( (_DWORD)v2 == 2 )
    {
      LODWORD(v5) = 0;
      BYTE4(v5) = 1;
      return v5;
    }
LABEL_8:
    if ( (_DWORD)v2 == 3 )
    {
      if ( v1[2] != 159 )
        goto LABEL_5;
    }
    else if ( (_DWORD)v2 == 6 && (v1[2] != 159 || v1[3] != 4096) )
    {
      goto LABEL_5;
    }
    goto LABEL_14;
  }
  if ( v3 != 16 )
  {
LABEL_5:
    BYTE4(v5) = 0;
    return v5;
  }
  if ( (_DWORD)v2 != 2 )
    goto LABEL_8;
LABEL_14:
  BYTE4(v5) = 1;
  LODWORD(v5) = v3 == 16;
  return v5;
}
