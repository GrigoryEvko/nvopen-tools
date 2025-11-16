// Function: sub_F6ED70
// Address: 0xf6ed70
//
__int64 __fastcall sub_F6ED70(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r15d
  __int64 v8; // rax
  unsigned int v10; // eax
  __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2;
  v5 = sub_F6C0B0(a1);
  if ( v5 )
  {
    v6 = v5;
    v7 = 0;
    if ( a2 )
    {
      v4 = a3;
      v7 = a3 * (a2 - 1);
      if ( **(_QWORD **)(a1 + 32) != *(_QWORD *)(v5 - 32) )
        goto LABEL_4;
    }
    else if ( **(_QWORD **)(a1 + 32) != *(_QWORD *)(v5 - 32) )
    {
LABEL_4:
      v11[0] = sub_BD5C60(v6);
      v8 = sub_B8C2F0(v11, v4, v7, 0);
      sub_B99FD0(v6, 2u, v8);
      return 1;
    }
    v10 = v7;
    v7 = v4;
    v4 = v10;
    goto LABEL_4;
  }
  return 0;
}
