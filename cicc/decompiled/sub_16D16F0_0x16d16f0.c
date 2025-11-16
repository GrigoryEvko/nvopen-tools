// Function: sub_16D16F0
// Address: 0x16d16f0
//
void __fastcall sub_16D16F0(unsigned __int8 *a1, int a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned __int8 *v5; // r12
  unsigned __int8 v6; // bl
  unsigned __int8 *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rdi
  char v10; // si
  _BYTE *v11; // rax
  unsigned __int8 v12; // bl
  char v13; // si
  _BYTE *v14; // rax

  if ( a2 )
  {
    v4 = (__int64)&a1[a2 - 1 + 1];
    v5 = a1;
    while ( 1 )
    {
      v6 = *v5;
      v7 = *(unsigned __int8 **)(a3 + 24);
      v8 = *(_QWORD *)(a3 + 16);
      if ( (unsigned __int8)(*v5 - 32) <= 0x5Eu && v6 != 92 && v6 != 34 )
        break;
      if ( (unsigned __int64)v7 >= v8 )
      {
        v9 = sub_16E7DE0(a3, 92);
      }
      else
      {
        v9 = a3;
        *(_QWORD *)(a3 + 24) = v7 + 1;
        *v7 = 92;
      }
      v10 = (v6 >> 4) + 55;
      v11 = *(_BYTE **)(v9 + 24);
      if ( (unsigned __int8)(v6 >> 4) <= 9u )
        v10 = (v6 >> 4) + 48;
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v9 + 16) )
      {
        v9 = sub_16E7DE0(v9, (unsigned int)v10);
      }
      else
      {
        *(_QWORD *)(v9 + 24) = v11 + 1;
        *v11 = v10;
      }
      v12 = v6 & 0xF;
      v13 = v12 + 55;
      if ( v12 <= 9u )
        v13 = v12 + 48;
      v14 = *(_BYTE **)(v9 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(v9 + 16) )
      {
        sub_16E7DE0(v9, (unsigned int)v13);
LABEL_7:
        if ( (unsigned __int8 *)v4 == ++v5 )
          return;
      }
      else
      {
        ++v5;
        *(_QWORD *)(v9 + 24) = v14 + 1;
        *v14 = v13;
        if ( (unsigned __int8 *)v4 == v5 )
          return;
      }
    }
    if ( (unsigned __int64)v7 >= v8 )
    {
      sub_16E7DE0(a3, v6);
    }
    else
    {
      *(_QWORD *)(a3 + 24) = v7 + 1;
      *v7 = v6;
    }
    goto LABEL_7;
  }
}
