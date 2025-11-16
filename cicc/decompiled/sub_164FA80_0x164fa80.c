// Function: sub_164FA80
// Address: 0x164fa80
//
void __fastcall sub_164FA80(__int64 *a1, __int64 a2)
{
  bool v4; // cc
  __int64 v5; // rsi
  __int64 v6; // rdi
  _BYTE *v7; // rax

  if ( a2 )
  {
    v4 = *(_BYTE *)(a2 + 16) <= 0x17u;
    v5 = *a1;
    if ( v4 )
    {
      sub_1553920((__int64 *)a2, v5, 1, (__int64)(a1 + 2));
      v6 = *a1;
      v7 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v7 < *(_QWORD *)(*a1 + 16) )
      {
LABEL_4:
        *(_QWORD *)(v6 + 24) = v7 + 1;
        *v7 = 10;
        return;
      }
    }
    else
    {
      sub_155BD40(a2, v5, (__int64)(a1 + 2), 0);
      v6 = *a1;
      v7 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v7 < *(_QWORD *)(*a1 + 16) )
        goto LABEL_4;
    }
    sub_16E7DE0(v6, 10);
  }
}
