// Function: sub_29BEBE0
// Address: 0x29bebe0
//
bool __fastcall sub_29BEBE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // rbx
  _QWORD *v10; // rdi
  int v11; // edx
  _QWORD *v12; // rax
  char v13; // al
  unsigned __int64 v14; // rax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v5 = a1 + 48;
  v6 = *(_QWORD *)(a1 + 56);
  if ( a1 + 48 != v6 )
  {
    do
    {
      v14 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v6 )
      {
        v10 = (_QWORD *)(v6 - 24);
        if ( v5 == v14 )
          goto LABEL_8;
      }
      else
      {
        if ( v5 == v14 )
          goto LABEL_9;
        v10 = 0;
      }
      if ( !v14 )
        BUG();
      v11 = *(unsigned __int8 *)(v14 - 24);
      v12 = (_QWORD *)(v14 - 24);
      if ( (unsigned int)(v11 - 30) >= 0xB )
        v12 = 0;
      if ( v12 != v10 )
      {
LABEL_8:
        v16 = a5;
        v13 = sub_29BDD80(v10, a2, a3, a4, a5, 1u);
        a5 = v16;
        if ( !v13 )
          return v6 == v5;
      }
LABEL_9:
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v5 != v6 );
  }
  return v6 == v5;
}
