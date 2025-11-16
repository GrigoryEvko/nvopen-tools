// Function: sub_2B15730
// Address: 0x2b15730
//
__int64 __fastcall sub_2B15730(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v3; // rdi
  __int64 v4; // rdx
  char v5; // cl
  __int64 v6; // [rsp+0h] [rbp-8h]

  if ( *(_BYTE *)a1 == 90 )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      v3 = *(_QWORD *)(a1 - 8);
    else
      v3 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v4 = *(_QWORD *)(v3 + 32);
    v5 = 0;
    if ( *(_BYTE *)v4 == 17 )
    {
      v1 = *(_QWORD **)(v4 + 24);
      if ( *(_DWORD *)(v4 + 32) > 0x40u )
        v1 = (_QWORD *)*v1;
      v5 = 1;
    }
    LODWORD(v6) = (_DWORD)v1;
    BYTE4(v6) = v5;
    return v6;
  }
  else
  {
    if ( *(_DWORD *)(a1 + 80) == 1 )
    {
      BYTE4(v6) = 1;
      LODWORD(v6) = **(_DWORD **)(a1 + 72);
    }
    else
    {
      BYTE4(v6) = 0;
    }
    return v6;
  }
}
