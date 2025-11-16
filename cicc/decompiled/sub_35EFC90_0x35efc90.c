// Function: sub_35EFC90
// Address: 0x35efc90
//
void __fastcall sub_35EFC90(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rax
  _QWORD *v8; // [rsp+8h] [rbp-28h]

  if ( *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) )
  {
    v7 = (_BYTE *)a4[4];
    if ( (_BYTE *)a4[3] == v7 )
    {
      v8 = a4;
      sub_CB6200((__int64)a4, (unsigned __int8 *)"+", 1u);
      a4 = v8;
    }
    else
    {
      *v7 = 43;
      ++a4[4];
    }
    sub_35EE840(a1, a2, a3, a4, a5, a6);
  }
}
