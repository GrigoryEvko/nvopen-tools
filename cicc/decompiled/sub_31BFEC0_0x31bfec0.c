// Function: sub_31BFEC0
// Address: 0x31bfec0
//
char __fastcall sub_31BFEC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // r15
  __int64 v10; // rdi
  unsigned __int8 v11; // r9
  __int64 v13; // [rsp+0h] [rbp-50h]
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  unsigned __int8 v16; // [rsp+18h] [rbp-38h]

  LOBYTE(v4) = a2 - 1;
  v6 = a2;
  v7 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v8 = (__int64 *)(a1 + 8 * v7);
      v9 = *(_QWORD *)(a4 + 8);
      v10 = *v8;
      v11 = (unsigned int)**(unsigned __int8 **)(*(_QWORD *)(*v8 + 8) + 16LL) - 30 <= 0xA;
      LOBYTE(v4) = (unsigned int)**(unsigned __int8 **)(v9 + 16) - 30 <= 0xA;
      if ( v11 == (_BYTE)v4 )
      {
        v13 = a4;
        v14 = a3;
        v15 = *(_QWORD *)(*v8 + 8);
        v16 = sub_318B700(v15);
        LOBYTE(v4) = sub_318B700(v9);
        a3 = v14;
        a4 = v13;
        if ( v16 == (_BYTE)v4 )
        {
          LOBYTE(v4) = sub_B445A0(*(_QWORD *)(v9 + 16), *(_QWORD *)(v15 + 16));
          a3 = v14;
          a4 = v13;
          if ( !(_BYTE)v4 )
            break;
        }
        else if ( v16 >= (unsigned __int8)v4 )
        {
          break;
        }
        v10 = *v8;
      }
      else if ( v11 <= (unsigned __int8)v4 )
      {
        break;
      }
      *(_QWORD *)(a1 + 8 * v6) = v10;
      v6 = v7;
      v4 = (v7 - 1) / 2;
      if ( a3 >= v7 )
        goto LABEL_12;
      v7 = (v7 - 1) / 2;
    }
  }
  v8 = (__int64 *)(a1 + 8 * v6);
LABEL_12:
  *v8 = a4;
  return v4;
}
