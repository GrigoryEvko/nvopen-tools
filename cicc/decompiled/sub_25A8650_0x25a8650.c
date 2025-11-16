// Function: sub_25A8650
// Address: 0x25a8650
//
void __fastcall sub_25A8650(unsigned __int64 a1, __int64 a2, _QWORD *a3, char a4)
{
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rax
  int v10; // edx
  _BYTE *v11; // rax
  _BYTE *i; // rdx

  v8 = (unsigned int)sub_B46E30(a2);
  v9 = a3[1];
  v10 = v8;
  if ( v8 == v9 )
    goto LABEL_9;
  if ( v8 >= v9 )
  {
    if ( v8 > a3[2] )
    {
      sub_C8D290((__int64)a3, a3 + 3, v8, 1u, v6, v7);
      v11 = (_BYTE *)(*a3 + a3[1]);
      for ( i = (_BYTE *)(v8 + *a3); i != v11; ++v11 )
      {
LABEL_5:
        if ( v11 )
          *v11 = 0;
      }
    }
    else
    {
      v11 = (_BYTE *)(*a3 + v9);
      i = (_BYTE *)(v8 + *a3);
      if ( v11 != i )
        goto LABEL_5;
    }
  }
  a3[1] = v8;
  v10 = sub_B46E30(a2);
LABEL_9:
  if ( v10 )
    sub_25A7990(a1, a2, (__int64)a3, a4, v6, v7);
}
