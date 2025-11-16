// Function: sub_183D8F0
// Address: 0x183d8f0
//
void __fastcall sub_183D8F0(unsigned __int64 a1, __int64 a2, _DWORD *a3, char a4)
{
  unsigned __int64 v6; // rdx
  int v7; // r8d
  int v8; // r9d
  unsigned __int64 v9; // rax
  int v10; // r15d
  int v11; // r8d
  int v12; // r9d
  _BYTE *v13; // rax
  _BYTE *i; // rdx
  unsigned __int64 v15; // [rsp+8h] [rbp-38h]

  v6 = (unsigned int)sub_15F4D60(a2);
  v9 = (unsigned int)a3[2];
  v10 = v6;
  if ( v6 >= v9 )
  {
    if ( v6 <= v9 )
    {
      if ( !(unsigned int)sub_15F4D60(a2) )
        return;
      goto LABEL_4;
    }
    if ( v6 > (unsigned int)a3[3] )
    {
      v15 = v6;
      sub_16CD150((__int64)a3, a3 + 4, v6, 1, v7, v8);
      v9 = (unsigned int)a3[2];
      v6 = v15;
    }
    v13 = (_BYTE *)(*(_QWORD *)a3 + v9);
    for ( i = (_BYTE *)(*(_QWORD *)a3 + v6); i != v13; ++v13 )
    {
      if ( v13 )
        *v13 = 0;
    }
  }
  a3[2] = v10;
  if ( (unsigned int)sub_15F4D60(a2) )
LABEL_4:
    sub_183CCB0(a1, a2, a3, a4, v11, v12);
}
